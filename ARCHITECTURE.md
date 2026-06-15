# Architecture

This document explains how the system works stage by stage, the responsibility of each module, and the reasoning behind the key design decisions. For setup and usage, see the [README](README.md).

---

## Two phases: Ingestion and Querying

The system splits cleanly into an **offline ingestion phase** (run once per document, expensive) and an **online query phase** (run per question, latency-sensitive). Separating them is what makes the demo fast: embeddings are pre-computed and shipped in `data/premade_embeddings/`, so querying never re-parses a PDF.

```
INGESTION (offline)                          QUERYING (online)
─────────────────────                        ──────────────────
PDF                                          Question
 │ Docling parse                              │ router (by question kind)
 ▼                                            ▼
structured JSON  ──► table serialization     vector retrieval (FAISS)
 │ page merge                                 │ LLM rerank (hybrid)
 ▼                                            │ parent-page expansion
page markdown ──► chunk ──► embed ──► FAISS   ▼
                                  └─ BM25      structured CoT answer
                                              │ citation validation
                                              ▼
                                              answer + page refs
```

---

## Ingestion pipeline

### 1. PDF parsing — `src/pdf_parsing.py`
[Docling](https://github.com/DS4SD/docling) converts each PDF into a structured JSON representation preserving text, reading order, layout, and **tables as first-class objects** (not flattened text). Parsing is parallelized across processes (`parse_and_export_parallel`) because Docling is CPU-bound and slow on large reports.

**Why Docling over PyPDF text extraction?** Financial reports are multi-column and table-heavy. Naïve text extraction scrambles reading order and destroys tables — the exact data users ask about. Docling keeps structure intact.

### 2. Table serialization — `src/tables_serialization.py`
Optional stage (`use_serialized_tables`). Each table is rewritten into a set of **independent, context-rich text blocks** so that a single row's meaning survives chunking and embedding. A figure like "Revenue 2023: €4.2M" stays semantically whole instead of being split from its header.

### 3. Page merging — `src/parsed_reports_merging.py`
Collapses the rich Docling JSON into a simpler **page-oriented structure**: a list of pages, each a single markdown string. This is the canonical form everything downstream consumes, and it preserves the **page number** — the unit of citation.

### 4. Chunking — `src/text_splitter.py`
`RecursiveCharacterTextSplitter.from_tiktoken_encoder`, **300 tokens / 50-token overlap**. Token-based (not character-based) splitting keeps chunks aligned with the embedding model's budget. Each chunk retains its parent `page`, which enables parent-document retrieval later.

### 5. Indexing — `src/ingestion.py`
- **`VectorDBIngestor`** embeds chunks with OpenAI `text-embedding-3-large` (3072-dim) and writes a FAISS `IndexFlatIP` per document. Inner-product on normalized vectors = cosine similarity. Embedding calls are retried with backoff (`tenacity`).
- **`BM25Ingestor`** (optional) builds a `BM25Okapi` lexical index per document, pickled to disk — useful for exact-term matches that dense retrieval can miss.

Indexes are keyed by the document's `sha1_name`, so each report has an isolated index. This is deliberate: queries are scoped to a single company/document (see retrieval), which keeps recall high and avoids cross-document contamination.

---

## Query pipeline

### 6. Routing — `src/questions_processing.py` + `src/prompts.py`
`QuestionsProcessor.process_question` first identifies which document(s) the question targets:
- **New-challenge mode** matches company names from `subset.csv` against the question text (longest-name-first to avoid partial-match collisions).
- A question naming **multiple companies** is dispatched to the comparative path (step 10).

The question's `kind` (`number`, `name`, `names`, `boolean`, `comparative`) selects a dedicated prompt class in `prompts.py`, each carrying its own **Pydantic answer schema**. Constraining the output shape per question type sharply reduces malformed answers.

### 7. Retrieval — `src/retrieval.py`
- **`VectorRetriever`** embeds the query, searches the target document's FAISS index, and returns the top-N chunks (or their parent pages).
- **`HybridRetriever`** wraps the vector retriever with an LLM reranking step.
- **`full_context`** mode skips retrieval entirely and feeds the whole document (used by long-context Gemini configs).

Retrieval is **always scoped to one document** via `retrieve_by_company_name`. The vector DB stores the chunk index; results are mapped back to chunk text and parent page.

### 8. LLM reranking — `src/reranking.py`
The reranker over-retrieves (e.g. top-28), then asks an LLM to score each candidate block's relevance to the query. The final ranking blends the vector similarity and the LLM relevance score:

```
combined = llm_weight * llm_score + (1 - llm_weight) * vector_score      # llm_weight ≈ 0.7
```

**Why rerank?** Dense vector similarity captures topical closeness but not always *answer-bearing* relevance. A cheap LLM pass on a small candidate set recovers precision without the cost of running the LLM over the whole corpus.

### 9. Parent-document retrieval
When enabled, retrieval matches on small chunks (high precision) but returns the **full parent page** (high context). This gives the answering model surrounding context — table headers, neighboring sentences — that a 300-token chunk alone would omit.

### 10. Answering — `src/questions_processing.py` + `src/api_requests.py`
The selected schema and the formatted RAG context go to the answering model. Every answer is structured chain-of-thought:

```jsonc
{
  "step_by_step_analysis": "…reasoning over the retrieved context…",
  "reasoning_summary": "…condensed rationale…",
  "relevant_pages": [12, 13],
  "final_answer": "…"
}
```

`api_requests.py` abstracts three providers behind one `send_message` interface:
- **OpenAI** uses native structured outputs (`response_format`).
- **IBM / Gemini** lack strict structured output, so responses are validated against the Pydantic schema and, on failure, **reparsed**: an `AnswerSchemaFixPrompt` re-asks the model to fix the JSON, then `json_repair` salvages the result before a final validation. This is the robustness layer that lets non-OpenAI models participate reliably.

### 11. Citation validation — `_validate_page_references`
A safeguard against hallucinated citations: every page the model claims is **intersected with the pages actually retrieved**. Invented references are dropped (and logged). If too few valid citations remain, top retrieved pages backfill to a minimum; an upper bound trims over-citation. For the competition's submission format, page indices are converted from 1-based (debug-friendly) to 0-based.

### 12. Comparative questions — `process_comparative_question`
Multi-company questions are decomposed: the LLM rephrases the question into one focused sub-question per company; each runs **in parallel** through the standard single-company path; references are de-duplicated and the per-company answers are synthesized into a final comparative answer.

---

## Concurrency model

- **PDF parsing:** process-level parallelism (CPU-bound Docling work).
- **Question processing:** `ThreadPoolExecutor` batches (I/O-bound LLM calls). Progress is checkpointed to disk after each batch via `_save_progress`, so a long run is resumable/inspectable. Shared answer-detail state is guarded by a `threading.Lock`.

---

## Provider abstraction

| Provider | Structured output | Embeddings | Notes |
|---|---|---|---|
| OpenAI | Native `response_format` | `text-embedding-3-large` | Primary path |
| Gemini | Validate + reparse | – | Long-context / "thinking" configs |
| IBM WatsonX | Validate + reparse | `granite-embedding` | Competition-only endpoint |

---

## Design decisions worth calling out

1. **Per-document indexes + question routing** instead of one global index — keeps retrieval precise and citations attributable to the right source.
2. **Rerank a small candidate set** rather than trust raw vector rank — precision gain at bounded cost.
3. **Schema-constrained structured outputs** per question kind — fewer malformed answers, machine-checkable results.
4. **Citation validation as a hard gate** — the model cannot cite a page it never saw, which is the difference between a demo and something trustworthy.
5. **Pre-computed embeddings shipped with the repo** — the demo is explorable in seconds without re-running ingestion.
