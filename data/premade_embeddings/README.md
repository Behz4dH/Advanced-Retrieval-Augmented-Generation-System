# Pre-made Embeddings

This directory contains pre-processed documents with ready-to-use embeddings to provide instant demo experience.

## Available Documents

### 1. Tradition Annual Report (test_tradition)
- **File**: `2779336b845a41544348abb7b3e6e5bd2ff893a2.pdf` from test_set
- **Type**: Financial annual report
- **Evaluation**: ✅ Available (has ground truth answers)
- **Features**: Numerical data, financial metrics, tables
- **Sample Questions**: Operating margin, revenue figures, financial performance

### 2. Harry Potter (harrypotter)
- **File**: `harrypotter.pdf`
- **Type**: Literary content
- **Evaluation**: ❌ Not available (no ground truth)
- **Features**: Narrative text, characters, plot
- **Sample Questions**: Character names, plot elements, story details

## Structure

Each pre-made document has:
```
<document_name>/
├── parsed_reports/     # Docling parsed JSON
├── merged_reports/     # Simplified page structure
├── chunked_reports/    # Text chunks for retrieval
└── vector_dbs/        # FAISS embeddings (ready to use)
```

## Usage

These embeddings are loaded directly in the Streamlit app when users select pre-made options, skipping the time-intensive embedding creation process.