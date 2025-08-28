import streamlit as st
import tempfile
import json
import os
from pathlib import Path
import time
import logging
from typing import Dict, List
import pandas as pd

# Import the RAG pipeline components
from src.pipeline import Pipeline, RunConfig
from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.retrieval import VectorRetriever, HybridRetriever
from src.reranking import LLMReranker
from src.prompts import AnswerWithRAGContextNamePrompt, AnswerWithRAGContextNumberPrompt
from openai import OpenAI
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'pipeline_step' not in st.session_state:
    st.session_state.pipeline_step = 0
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if 'document_type' not in st.session_state:
    st.session_state.document_type = "upload"  # "upload", "premade"
if 'selected_premade' not in st.session_state:
    st.session_state.selected_premade = None
if 'advanced_settings' not in st.session_state:
    st.session_state.advanced_settings = {
        'enable_reranking': True,
        'enable_parent_retrieval': True,
        'enable_cot': True,
        'top_n_retrieval': 10,
        'reranking_sample_size': 20,
        'llm_weight': 0.7,
        'answering_model': 'gpt-4o-mini-2024-07-18'
    }

def load_premade_documents():
    """Load information about available pre-made documents"""
    premade_dir = Path("data/premade_embeddings")
    if not premade_dir.exists():
        return {}
    
    documents = {}
    for doc_dir in premade_dir.iterdir():
        if doc_dir.is_dir() and (doc_dir / "document_info.json").exists():
            try:
                with open(doc_dir / "document_info.json", 'r', encoding='utf-8') as f:
                    doc_info = json.load(f)
                    documents[doc_info['doc_id']] = doc_info
            except Exception as e:
                st.warning(f"Could not load document info for {doc_dir.name}: {e}")
    
    return documents

def setup_premade_document(doc_id: str, doc_info: dict) -> Dict[Path, Path]:
    """Setup directory structure for pre-made document"""
    premade_dir = Path("data/premade_embeddings") / doc_id
    
    dirs = {
        'root': premade_dir,
        'pdf_dir': premade_dir / "pdf_reports",
        'parsed_dir': premade_dir / "debug_data" / "01_parsed_reports", 
        'merged_dir': premade_dir / "debug_data" / "02_merged_reports",
        'chunked_dir': premade_dir / "databases" / "chunked_reports",
        'vector_db_dir': premade_dir / "databases" / "vector_dbs"
    }
    
    # Verify all required directories exist
    required_dirs = ['chunked_dir', 'vector_db_dir']
    for dir_key in required_dirs:
        if not dirs[dir_key].exists():
            st.error(f"❌ Pre-made embeddings not found for {doc_info['name']}. Please run create_premade_embeddings.py first.")
            return None
    
    return dirs

def has_evaluation_available(doc_id: str) -> bool:
    """Check if evaluation is available for this document"""
    premade_docs = load_premade_documents()
    if doc_id in premade_docs:
        return premade_docs[doc_id].get('has_evaluation', False)
    return False

def create_temp_directories():
    """Create temporary directories for processing"""
    temp_dir = Path(tempfile.mkdtemp())
    
    dirs = {
        'root': temp_dir,
        'pdf_dir': temp_dir / "pdf_reports",
        'parsed_dir': temp_dir / "parsed_reports", 
        'merged_dir': temp_dir / "merged_reports",
        'chunked_dir': temp_dir / "chunked_reports",
        'vector_db_dir': temp_dir / "vector_dbs"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def save_uploaded_pdf(uploaded_file, pdf_dir: Path):
    """Save uploaded PDF to temp directory"""
    pdf_path = pdf_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return pdf_path

def create_dummy_csv(csv_path: Path, pdf_filename: str):
    """Create a dummy CSV metadata file for the uploaded PDF"""
    # Extract filename without extension to use as SHA1 name
    sha1_name = Path(pdf_filename).stem
    
    # Create a simple CSV with minimal required fields
    df = pd.DataFrame({
        'sha1': [sha1_name],
        'company_name': ['Demo Company']
    })
    df.to_csv(csv_path, index=False)
    return sha1_name

def step1_parse_pdf(pdf_path: Path, dirs: Dict[Path, Path]) -> Dict:
    """Step 1: Parse PDF using Docling"""
    st.subheader("🔄 Step 1: PDF Parsing")
    
    with st.spinner("Parsing PDF with Docling..."):
        # Create dummy CSV for metadata
        csv_path = dirs['root'] / "subset.csv"
        sha1_name = create_dummy_csv(csv_path, pdf_path.name)
        
        # Initialize PDF parser
        parser = PDFParser(
            output_dir=dirs['parsed_dir'],
            csv_metadata_path=csv_path
        )
        
        try:
            # Parse the PDF
            start_time = time.time()
            parser.parse_and_export(input_doc_paths=[pdf_path])
            processing_time = time.time() - start_time
            
            # Load the parsed result
            parsed_file = dirs['parsed_dir'] / f"{sha1_name}.json"
            with open(parsed_file, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
                
            st.success(f"✅ PDF parsed successfully in {processing_time:.2f}s")
            
            # Show parsing results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages", parsed_data['metainfo']['pages_amount'])
                st.metric("Text Blocks", parsed_data['metainfo']['text_blocks_amount'])
            with col2:
                st.metric("Tables", parsed_data['metainfo']['tables_amount'])
                st.metric("Pictures", parsed_data['metainfo']['pictures_amount'])
            
            # Show sample content
            if parsed_data['content'] and len(parsed_data['content']) > 0:
                st.write("**Sample Content from First Page:**")
                first_page = parsed_data['content'][0]
                if 'content' in first_page and len(first_page['content']) > 0:
                    sample_text = first_page['content'][0].get('text', 'No text found')[:500]
                    st.text_area("Sample Text", sample_text, height=100, disabled=True)
            
            return {
                'parsed_data': parsed_data,
                'sha1_name': sha1_name,
                'processing_time': processing_time
            }
            
        except Exception as e:
            st.error(f"❌ Error parsing PDF: {str(e)}")
            return None

def step2_merge_reports(dirs: Dict[Path, Path], sha1_name: str) -> Dict:
    """Step 2: Convert to simpler JSON structure"""
    st.subheader("🔧 Step 2: Report Processing")
    
    with st.spinner("Converting to simpler format..."):
        try:
            ptp = PageTextPreparation(use_serialized_tables=False)
            start_time = time.time()
            
            # Process the parsed report
            processed_reports = ptp.process_reports(
                reports_dir=dirs['parsed_dir'],
                output_dir=dirs['merged_dir']
            )
            
            processing_time = time.time() - start_time
            
            # Load the merged result
            merged_file = dirs['merged_dir'] / f"{sha1_name}.json"
            with open(merged_file, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            st.success(f"✅ Report processed in {processing_time:.2f}s")
            
            # Show processing results
            pages_count = len(merged_data['content']['pages'])
            st.metric("Pages Processed", pages_count)
            
            # Show sample of merged content
            if merged_data['content']['pages']:
                st.write("**Sample Page Content:**")
                first_page = merged_data['content']['pages'][0]
                sample_text = first_page.get('text', 'No text found')[:400]
                st.text_area("Merged Text", sample_text, height=100, disabled=True)
            
            return {
                'merged_data': merged_data,
                'processing_time': processing_time,
                'pages_count': pages_count
            }
            
        except Exception as e:
            st.error(f"❌ Error processing report: {str(e)}")
            return None

def step3_chunk_text(dirs: Dict[Path, Path], sha1_name: str) -> Dict:
    """Step 3: Split text into chunks"""
    st.subheader("✂️ Step 3: Text Chunking")
    
    with st.spinner("Splitting text into chunks..."):
        try:
            text_splitter = TextSplitter()
            start_time = time.time()
            
            # Split the reports into chunks
            text_splitter.split_all_reports(
                dirs['merged_dir'],
                dirs['chunked_dir']
            )
            
            processing_time = time.time() - start_time
            
            # Load the chunked result
            chunked_file = dirs['chunked_dir'] / f"{sha1_name}.json"
            with open(chunked_file, 'r', encoding='utf-8') as f:
                chunked_data = json.load(f)
            
            chunks = chunked_data['content']['chunks']
            total_chunks = len(chunks)
            
            st.success(f"✅ Text chunked in {processing_time:.2f}s")
            
            # Show chunking results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", total_chunks)
            with col2:
                avg_tokens = sum(chunk.get('length_tokens', 0) for chunk in chunks) / total_chunks if chunks else 0
                st.metric("Avg Tokens/Chunk", f"{avg_tokens:.0f}")
            
            # Show sample chunks
            st.write("**Sample Chunks:**")
            for i, chunk in enumerate(chunks[:3]):
                with st.expander(f"Chunk {i+1} (Page {chunk.get('page', 'N/A')}) - {chunk.get('length_tokens', 0)} tokens"):
                    st.text(chunk.get('text', 'No text')[:300] + "..." if len(chunk.get('text', '')) > 300 else chunk.get('text', 'No text'))
            
            return {
                'chunked_data': chunked_data,
                'processing_time': processing_time,
                'total_chunks': total_chunks
            }
            
        except Exception as e:
            st.error(f"❌ Error chunking text: {str(e)}")
            return None

def step4_create_vector_db(dirs: Dict[Path, Path], sha1_name: str) -> Dict:
    """Step 4: Create vector database"""
    st.subheader("🗄️ Step 4: Vector Database Creation")
    
    # Check if OpenAI API key is available
    if not st.session_state.openai_api_key:
        st.error("❌ OpenAI API key not found. Please enter your API key in the sidebar.")
        return None
    
    # Set the API key for this process
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    
    with st.spinner("Creating vector embeddings..."):
        try:
            vdb_ingestor = VectorDBIngestor()
            start_time = time.time()
            
            # Create vector database
            vdb_ingestor.process_reports(dirs['chunked_dir'], dirs['vector_db_dir'])
            
            processing_time = time.time() - start_time
            
            # Check if vector DB was created
            vector_db_file = dirs['vector_db_dir'] / f"{sha1_name}.faiss"
            if not vector_db_file.exists():
                st.error("❌ Vector database file was not created")
                return None
            
            st.success(f"✅ Vector database created in {processing_time:.2f}s")
            
            # Show vector DB info
            st.metric("Vector DB File Size", f"{vector_db_file.stat().st_size / 1024:.1f} KB")
            st.info("Vector database uses OpenAI's text-embedding-3-large model with 3072 dimensions")
            
            return {
                'processing_time': processing_time,
                'vector_db_path': vector_db_file
            }
            
        except Exception as e:
            st.error(f"❌ Error creating vector database: {str(e)}")
            return None

def get_answer_with_cot(query: str, context: str, answering_model: str = "gpt-4o-mini-2024-07-18") -> Dict:
    """Generate Chain of Thought answer using structured prompts"""
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Determine if this is a numerical question
    numerical_patterns = [r'\bhow much\b', r'\bhow many\b', r'\bwhat.*amount\b', r'\bwhat.*cost\b', 
                         r'\bwhat.*price\b', r'\bwhat.*revenue\b', r'\bwhat.*percent\b', r'\d+']
    is_numerical = any(re.search(pattern, query.lower()) for pattern in numerical_patterns)
    
    # Select appropriate prompt and schema
    if is_numerical:
        prompt_class = AnswerWithRAGContextNumberPrompt
        schema = prompt_class.AnswerSchema
    else:
        prompt_class = AnswerWithRAGContextNamePrompt
        schema = prompt_class.AnswerSchema
    
    # Format the user prompt
    user_prompt = prompt_class.user_prompt.format(context=context, question=query)
    
    try:
        completion = client.beta.chat.completions.parse(
            model=answering_model,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt_class.system_prompt_with_schema},
                {"role": "user", "content": user_prompt}
            ],
            response_format=schema
        )
        
        response = completion.choices[0].message.parsed
        return response.model_dump()
        
    except Exception as e:
        return {
            "step_by_step_analysis": f"Error generating Chain of Thought response: {str(e)}",
            "reasoning_summary": "Analysis failed due to error",
            "relevant_pages": [],
            "final_answer": "N/A"
        }

def step5_query_system(dirs: Dict[Path, Path], sha1_name: str, in_tab: bool = False, company_name: str = "Demo Company") -> Dict:
    """Step 5: Advanced Query & Retrieval with CoT"""
    if not in_tab:
        st.subheader("🔍 Step 5: Advanced Query & Retrieval")
    
    # Check if we have the necessary files
    if not st.session_state.openai_api_key:
        st.error("❌ OpenAI API key not found for querying.")
        return None
    
    # Set the API key for this process
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    
    try:
        # Initialize retrievers based on settings
        vector_retriever = VectorRetriever(
            vector_db_dir=dirs['vector_db_dir'],
            documents_dir=dirs['chunked_dir']
        )
        
        retriever = vector_retriever
        if st.session_state.advanced_settings['enable_reranking']:
            retriever = HybridRetriever(
                vector_db_dir=dirs['vector_db_dir'],
                documents_dir=dirs['chunked_dir']
            )
        
        st.success("✅ Advanced retrieval system initialized")
        
        # Advanced settings display
        with st.expander("⚙️ Current Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**LLM Reranking:** {'✅' if st.session_state.advanced_settings['enable_reranking'] else '❌'}")
                st.write(f"**Parent Retrieval:** {'✅' if st.session_state.advanced_settings['enable_parent_retrieval'] else '❌'}")
                st.write(f"**Chain of Thought:** {'✅' if st.session_state.advanced_settings['enable_cot'] else '❌'}")
            with col2:
                st.write(f"**Top N Results:** {st.session_state.advanced_settings['top_n_retrieval']}")
                st.write(f"**Reranking Sample:** {st.session_state.advanced_settings['reranking_sample_size']}")
                st.write(f"**Model:** {st.session_state.advanced_settings['answering_model'][:20]}...")
        
        # Query interface
        st.write("**Ask questions about your document:**")
        
        # Sample queries
        sample_queries = [
            "What is the main topic of this document?",
            "What are the key findings mentioned?",
            "Tell me about any financial information",
            "How much revenue was generated?",
            "What recommendations are provided?"
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query_key = "tab_query_input" if in_tab else "main_query_input"
            query = st.text_input("Enter your question:", placeholder="What would you like to know about this document?", key=query_key)
        with col2:
            st.write("**Sample queries:**")
            for sample in sample_queries:
                button_key = f"{'tab_' if in_tab else ''}sample_{sample[:20]}"
                if st.button(sample, key=button_key):
                    query = sample
        
        search_button_key = f"{'tab_' if in_tab else ''}advanced_search_btn"
        if query and st.button("🔍 Advanced Search", type="primary", key=search_button_key):
            with st.spinner("Performing advanced search..."):
                try:
                    # Performance tracking
                    start_time = time.time()
                    retrieval_start = time.time()
                    
                    # Perform retrieval based on settings
                    if st.session_state.advanced_settings['enable_reranking'] and isinstance(retriever, HybridRetriever):
                        results = retriever.retrieve_and_rerank(
                            company_name=company_name,
                            query=query,
                            top_n=st.session_state.advanced_settings['top_n_retrieval'],
                            llm_reranking_sample_size=st.session_state.advanced_settings['reranking_sample_size'],
                            return_parent_pages=st.session_state.advanced_settings['enable_parent_retrieval'],
                            llm_weight=st.session_state.advanced_settings['llm_weight']
                        )
                    else:
                        results = retriever.retrieve_by_company_name(
                            company_name=company_name,
                            query=query,
                            top_n=st.session_state.advanced_settings['top_n_retrieval'],
                            return_parent_pages=st.session_state.advanced_settings['enable_parent_retrieval']
                        )
                    
                    retrieval_time = time.time() - retrieval_start
                    
                    # Generate Chain of Thought answer if enabled
                    cot_answer = None
                    cot_time = 0
                    if st.session_state.advanced_settings['enable_cot'] and results:
                        cot_start = time.time()
                        context = "\n\n".join([f"Page {r['page']}:\n{r['text']}" for r in results])
                        cot_answer = get_answer_with_cot(query, context, st.session_state.advanced_settings['answering_model'])
                        cot_time = time.time() - cot_start
                    
                    total_time = time.time() - start_time
                    
                    st.success(f"✅ Advanced search completed in {total_time:.2f}s")
                    
                    # Performance Metrics
                    with st.expander("📊 Performance Metrics", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Time", f"{total_time:.3f}s")
                        with col2:
                            st.metric("Retrieval Time", f"{retrieval_time:.3f}s")
                        with col3:
                            st.metric("CoT Time", f"{cot_time:.3f}s" if cot_time > 0 else "N/A")
                        with col4:
                            st.metric("Results Found", len(results))
                    
                    # Chain of Thought Answer (if enabled)
                    if cot_answer and st.session_state.advanced_settings['enable_cot']:
                        st.markdown("---")
                        st.subheader("🧠 Chain of Thought Analysis")
                        
                        # Structured output display
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write("**Final Answer:**")
                            answer_value = cot_answer.get('final_answer', 'N/A')
                            if answer_value != 'N/A':
                                st.success(f"✅ {answer_value}")
                            else:
                                st.warning("⚠️ No definitive answer found in document")
                        
                        with col2:
                            st.write("**Relevant Pages:**")
                            relevant_pages = cot_answer.get('relevant_pages', [])
                            if relevant_pages:
                                st.info(f"Pages: {', '.join(map(str, relevant_pages))}")
                            else:
                                st.warning("No specific pages identified")
                        
                        # Reasoning analysis
                        with st.expander("🔍 Detailed Analysis", expanded=True):
                            st.write("**Step-by-Step Analysis:**")
                            st.write(cot_answer.get('step_by_step_analysis', 'No analysis available'))
                            
                            st.write("**Reasoning Summary:**")
                            st.info(cot_answer.get('reasoning_summary', 'No summary available'))
                    
                    # Display retrieval results
                    st.markdown("---")
                    st.write(f"**Retrieved Passages ({len(results)} found):**")
                    
                    for i, result in enumerate(results):
                        # Enhanced result display with reranking scores
                        similarity_score = result['distance']
                        page_num = result['page']
                        text = result['text']
                        
                        # Additional scores if reranking was used
                        relevance_score = result.get('relevance_score', None)
                        combined_score = result.get('combined_score', None)
                        
                        title = f"Passage {i+1} - Page {page_num}"
                        if relevance_score is not None:
                            title += f" (Vector: {similarity_score:.3f}, LLM: {relevance_score:.3f}, Combined: {combined_score:.3f})"
                        else:
                            title += f" (Similarity: {similarity_score:.3f})"
                        
                        with st.expander(title):
                            st.write(text)
                            
                            # Show reranking details if available
                            if relevance_score is not None:
                                st.caption(f"🔄 Reranked by LLM | Vector Score: {similarity_score:.3f} | LLM Relevance: {relevance_score:.3f}")
                    
                    return {
                        'query': query,
                        'results': results,
                        'total_time': total_time,
                        'retrieval_time': retrieval_time,
                        'cot_time': cot_time,
                        'cot_answer': cot_answer
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error during advanced search: {str(e)}")
                    st.exception(e)  # Show detailed error for debugging
                    return None
        
        return {'initialized': True}
        
    except Exception as e:
        st.error(f"❌ Error initializing advanced retrieval system: {str(e)}")
        st.exception(e)  # Show detailed error for debugging
        return None

def evaluate_rag_system(dirs: Dict[Path, Path], sha1_name: str, company_name: str = "Demo Company") -> Dict:
    """Step 6: Evaluate RAG system performance against test questions"""
    st.subheader("📊 Step 6: RAG System Evaluation")
    
    if not st.session_state.openai_api_key:
        st.error("❌ OpenAI API key required for evaluation")
        return None
    
    # Set the API key
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    
    try:
        # Load test questions - try to find them relative to current directory
        project_root = Path.cwd()
        test_questions_path = project_root / "data" / "test_set" / "questions.json"
        ground_truth_path = project_root / "data" / "test_set" / "answers_max_nst_o3m.json"
        
        # Fallback: try different possible locations
        if not test_questions_path.exists():
            possible_paths = [
                Path("data/test_set/questions.json"),
                Path("./data/test_set/questions.json"),
                Path("../data/test_set/questions.json")
            ]
            for path in possible_paths:
                if path.exists():
                    test_questions_path = path
                    ground_truth_path = path.parent / "answers_max_nst_o3m.json"
                    break
        
        # Use custom Tradition-specific questions if testing Tradition document
        if company_name == "Tradition Annual Report":
            st.info("📊 Using Tradition-specific evaluation questions")
            test_questions = [
                {"text": "Did Tradition announce a dividend payment in the annual report?", "kind": "boolean"},
                {"text": "According to the annual report, what is the Operating margin (%) for Tradition (within the last period or at the end of the last period)? If data is not available, return 'N/A'.", "kind": "number"},
                {"text": "Did Tradition mention any acquisitions in the annual report?", "kind": "boolean"},
                {"text": "What was the dividend amount per share proposed by Tradition in CHF?", "kind": "number"},
                {"text": "What was Tradition's consolidated revenue in CHF millions for 2022?", "kind": "number"}
            ]
            
            # Ground truth for Tradition-specific questions
            ground_truth = {
                "Did Tradition announce a dividend payment in the annual report?": {
                    "value": True,
                    "kind": "boolean"
                },
                "According to the annual report, what is the Operating margin (%) for Tradition (within the last period or at the end of the last period)? If data is not available, return 'N/A'.": {
                    "value": 9.9,
                    "kind": "number"
                },
                "Did Tradition mention any acquisitions in the annual report?": {
                    "value": True,
                    "kind": "boolean"
                },
                "What was the dividend amount per share proposed by Tradition in CHF?": {
                    "value": 5.5,
                    "kind": "number"
                },
                "What was Tradition's consolidated revenue in CHF millions for 2022?": {
                    "value": 1028.6,
                    "kind": "number"
                }
            }
            
        elif not test_questions_path.exists():
            st.warning("⚠️ Test questions not found. Using sample evaluation questions.")
            test_questions = [
                {"text": "What is the main topic of this document?", "kind": "name"},
                {"text": "Are there any financial figures mentioned?", "kind": "boolean"},
                {"text": "What is the revenue amount if available?", "kind": "number"}
            ]
            ground_truth = None
        else:
            with open(test_questions_path, 'r', encoding='utf-8') as f:
                test_questions = json.load(f)
            
            ground_truth = None
            if ground_truth_path.exists():
                with open(ground_truth_path, 'r', encoding='utf-8') as f:
                    ground_truth_data = json.load(f)
                    ground_truth = {q["question_text"]: q for q in ground_truth_data["answers"]}

        # Initialize retrievers
        vector_retriever = VectorRetriever(
            vector_db_dir=dirs['vector_db_dir'],
            documents_dir=dirs['chunked_dir']
        )
        
        retriever = vector_retriever
        if st.session_state.advanced_settings['enable_reranking']:
            retriever = HybridRetriever(
                vector_db_dir=dirs['vector_db_dir'],
                documents_dir=dirs['chunked_dir']
            )

        st.write(f"**Evaluating against {len(test_questions)} test questions...**")
        
        if st.button("🚀 Start Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                evaluation_results = []
                total_start_time = time.time()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, question in enumerate(test_questions):
                    status_text.text(f"Processing question {i+1}/{len(test_questions)}")
                    progress_bar.progress((i + 1) / len(test_questions))
                    
                    question_text = question["text"]
                    question_kind = question["kind"]
                    
                    # Performance tracking
                    start_time = time.time()
                    
                    # Perform retrieval
                    try:
                        if st.session_state.advanced_settings['enable_reranking'] and isinstance(retriever, HybridRetriever):
                            results = retriever.retrieve_and_rerank(
                                company_name=company_name,
                                query=question_text,
                                top_n=st.session_state.advanced_settings['top_n_retrieval'],
                                llm_reranking_sample_size=st.session_state.advanced_settings['reranking_sample_size'],
                                return_parent_pages=st.session_state.advanced_settings['enable_parent_retrieval'],
                                llm_weight=st.session_state.advanced_settings['llm_weight']
                            )
                        else:
                            results = retriever.retrieve_by_company_name(
                                company_name=company_name,
                                query=question_text,
                                top_n=st.session_state.advanced_settings['top_n_retrieval'],
                                return_parent_pages=st.session_state.advanced_settings['enable_parent_retrieval']
                            )
                        
                        # Generate Chain of Thought answer if enabled
                        cot_answer = None
                        if st.session_state.advanced_settings['enable_cot'] and results:
                            context = "\n\n".join([f"Page {r['page']}:\n{r['text']}" for r in results])
                            cot_answer = get_answer_with_cot(question_text, context, st.session_state.advanced_settings['answering_model'])
                        
                        processing_time = time.time() - start_time
                        
                        # Evaluate answer if ground truth available
                        accuracy_score = None
                        if ground_truth and question_text in ground_truth:
                            gt_answer = ground_truth[question_text]
                            if cot_answer:
                                predicted_value = cot_answer.get('final_answer', 'N/A')
                                gt_value = gt_answer.get('value', 'N/A')
                                
                                if question_kind == "boolean":
                                    # Handle boolean comparisons more flexibly
                                    pred_bool = str(predicted_value).lower().strip()
                                    gt_bool = str(gt_value).lower().strip()
                                    
                                    # Convert various representations to standard bool
                                    true_values = ['true', 'yes', '1', 'correct', 'positive']
                                    false_values = ['false', 'no', '0', 'incorrect', 'negative']
                                    
                                    pred_is_true = pred_bool in true_values
                                    gt_is_true = gt_bool in true_values or gt_value is True
                                    
                                    accuracy_score = 1.0 if pred_is_true == gt_is_true else 0.0
                                    
                                elif question_kind == "number":
                                    try:
                                        # Handle percentage and number extraction
                                        pred_str = str(predicted_value).strip().replace('%', '')
                                        pred_num = float(pred_str) if pred_str != 'N/A' else None
                                        gt_num = float(gt_value) if gt_value != 'N/A' else None
                                        
                                        if pred_num is not None and gt_num is not None:
                                            accuracy_score = 1.0 if abs(pred_num - gt_num) < 0.01 else 0.0
                                        else:
                                            accuracy_score = 1.0 if pred_num == gt_num else 0.0
                                    except Exception as e:
                                        accuracy_score = 0.0
                                        
                                else:  # name/text
                                    accuracy_score = 1.0 if str(predicted_value).lower().strip() == str(gt_value).lower().strip() else 0.0
                        
                        evaluation_results.append({
                            'question': question_text,
                            'kind': question_kind,
                            'processing_time': processing_time,
                            'num_results': len(results),
                            'cot_answer': cot_answer,
                            'accuracy_score': accuracy_score,
                            'results': results[:3]  # Store top 3 results for analysis
                        })
                        
                    except Exception as e:
                        evaluation_results.append({
                            'question': question_text,
                            'kind': question_kind,
                            'processing_time': 0,
                            'num_results': 0,
                            'cot_answer': None,
                            'accuracy_score': 0.0,
                            'error': str(e),
                            'results': []
                        })
                
                total_time = time.time() - total_start_time
                progress_bar.progress(1.0)
                status_text.text("Evaluation completed!")
                
                # Calculate summary metrics
                successful_queries = [r for r in evaluation_results if 'error' not in r]
                failed_queries = [r for r in evaluation_results if 'error' in r]
                
                avg_processing_time = sum(r['processing_time'] for r in successful_queries) / len(successful_queries) if successful_queries else 0
                avg_results_count = sum(r['num_results'] for r in successful_queries) / len(successful_queries) if successful_queries else 0
                
                # Calculate accuracy if ground truth available
                accuracy_scores = [r['accuracy_score'] for r in evaluation_results if r['accuracy_score'] is not None]
                overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None
                
                st.success(f"✅ Evaluation completed in {total_time:.2f}s")
                
                # Display summary metrics
                st.markdown("---")
                st.subheader("📈 Evaluation Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Questions", len(test_questions))
                with col2:
                    st.metric("Successful", len(successful_queries))
                with col3:
                    st.metric("Failed", len(failed_queries))
                with col4:
                    if overall_accuracy is not None:
                        st.metric("Accuracy", f"{overall_accuracy:.2%}")
                    else:
                        st.metric("Accuracy", "N/A")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Time", f"{total_time:.2f}s")
                with col2:
                    st.metric("Avg Time/Question", f"{avg_processing_time:.3f}s")
                with col3:
                    st.metric("Throughput", f"{len(successful_queries)/total_time:.1f} Q/s")
                with col4:
                    st.metric("Avg Results/Query", f"{avg_results_count:.1f}")
                
                # Display detailed results
                st.markdown("---")
                st.subheader("📋 Detailed Results")
                
                for i, result in enumerate(evaluation_results):
                    with st.expander(f"Question {i+1}: {result['question'][:60]}... ({'✅' if 'error' not in result else '❌'})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Question:** {result['question']}")
                            st.write(f"**Type:** {result['kind']}")
                            
                            if 'error' in result:
                                st.error(f"**Error:** {result['error']}")
                            else:
                                if result['cot_answer']:
                                    st.write(f"**Answer:** {result['cot_answer'].get('final_answer', 'N/A')}")
                                    if result['accuracy_score'] is not None:
                                        accuracy_emoji = "✅" if result['accuracy_score'] == 1.0 else "❌"
                                        st.write(f"**Accuracy:** {accuracy_emoji} {result['accuracy_score']:.2%}")
                                else:
                                    st.write("**Answer:** No CoT answer generated")
                        
                        with col2:
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            st.metric("Results Found", result['num_results'])
                        
                        # Show reasoning if available
                        if result.get('cot_answer'):
                            st.write("**🧠 Chain of Thought Reasoning:**")
                            st.write("*Step-by-Step Analysis:*")
                            st.write(result['cot_answer'].get('step_by_step_analysis', 'No analysis available'))
                        
                        # Show top retrieval results
                        if result['results']:
                            st.write(f"**🔍 Top Retrieval Results ({len(result['results'])}):**")
                            for j, res in enumerate(result['results']):
                                st.write(f"**Result {j+1}** (Page {res['page']}, Score: {res['distance']:.3f})")
                                with st.container():
                                    st.text_area(f"Content {j+1}", res['text'][:300] + "..." if len(res['text']) > 300 else res['text'], 
                                                height=100, disabled=True, key=f"result_{i}_{j}")
                                if j < len(result['results']) - 1:
                                    st.markdown("---")
                
                return {
                    'total_questions': len(test_questions),
                    'successful_queries': len(successful_queries),
                    'failed_queries': len(failed_queries),
                    'overall_accuracy': overall_accuracy,
                    'total_time': total_time,
                    'avg_processing_time': avg_processing_time,
                    'throughput': len(successful_queries)/total_time,
                    'detailed_results': evaluation_results
                }
        
        return {'initialized': True}
        
    except Exception as e:
        st.error(f"❌ Error during evaluation: {str(e)}")
        st.exception(e)
        return None

def main():
    st.title("🔍 Advanced RAG Pipeline Demo")
    st.markdown("**Upload a PDF and watch the complete RAG process in action, plus evaluate system performance!**")
    
    # Sidebar with API key input and process overview
    st.sidebar.title("🔧 Configuration")
    
    # OpenAI API Key Input
    api_key_input = st.sidebar.text_input(
        "🔑 OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enter your OpenAI API key to enable embedding creation and querying"
    )
    
    if api_key_input != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key_input
        os.environ["OPENAI_API_KEY"] = api_key_input
    
    # Show API key status
    if st.session_state.openai_api_key:
        st.sidebar.success("✅ API Key Set")
    else:
        st.sidebar.warning("⚠️ API Key Required")
        st.sidebar.markdown("You need an OpenAI API key for steps 4 and 5 (Vector DB creation and querying)")
    
    st.sidebar.markdown("---")
    
    # Advanced Settings Panel
    st.sidebar.title("⚙️ Advanced Settings")
    
    # RAG Enhancement Options
    st.sidebar.subheader("🚀 RAG Enhancements")
    st.session_state.advanced_settings['enable_reranking'] = st.sidebar.checkbox(
        "🔄 LLM Reranking",
        value=st.session_state.advanced_settings['enable_reranking'],
        help="Use LLM to rerank retrieved passages for better relevance"
    )
    
    st.session_state.advanced_settings['enable_parent_retrieval'] = st.sidebar.checkbox(
        "📜 Parent Document Retrieval",
        value=st.session_state.advanced_settings['enable_parent_retrieval'],
        help="Retrieve full page context around relevant chunks"
    )
    
    st.session_state.advanced_settings['enable_cot'] = st.sidebar.checkbox(
        "🧠 Chain of Thought",
        value=st.session_state.advanced_settings['enable_cot'],
        help="Generate structured reasoning with step-by-step analysis"
    )
    
    # Retrieval Parameters
    st.sidebar.subheader("🎯 Retrieval Parameters")
    st.session_state.advanced_settings['top_n_retrieval'] = st.sidebar.slider(
        "Top N Results",
        min_value=3,
        max_value=20,
        value=st.session_state.advanced_settings['top_n_retrieval'],
        help="Number of final results to return"
    )
    
    if st.session_state.advanced_settings['enable_reranking']:
        st.session_state.advanced_settings['reranking_sample_size'] = st.sidebar.slider(
            "Reranking Sample Size",
            min_value=10,
            max_value=50,
            value=st.session_state.advanced_settings['reranking_sample_size'],
            help="Number of candidates to rerank (should be > Top N)"
        )
        
        st.session_state.advanced_settings['llm_weight'] = st.sidebar.slider(
            "LLM Weight",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.advanced_settings['llm_weight'],
            step=0.1,
            help="Weight of LLM score vs vector similarity (0.1=more vector, 1.0=more LLM)"
        )
    
    # Model Selection
    if st.session_state.advanced_settings['enable_cot']:
        st.sidebar.subheader("🤖 Language Model")
        model_options = [
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
            "gpt-3.5-turbo"
        ]
        st.session_state.advanced_settings['answering_model'] = st.sidebar.selectbox(
            "Answering Model",
            options=model_options,
            index=model_options.index(st.session_state.advanced_settings['answering_model']) if st.session_state.advanced_settings['answering_model'] in model_options else 0,
            help="Model used for Chain of Thought reasoning"
        )
    
    st.sidebar.markdown("---")
    
    # Settings Summary
    with st.sidebar.expander("📊 Current Configuration", expanded=False):
        st.write(f"**Reranking:** {'✅' if st.session_state.advanced_settings['enable_reranking'] else '❌'}")
        st.write(f"**Parent Retrieval:** {'✅' if st.session_state.advanced_settings['enable_parent_retrieval'] else '❌'}")
        st.write(f"**Chain of Thought:** {'✅' if st.session_state.advanced_settings['enable_cot'] else '❌'}")
        st.write(f"**Top N:** {st.session_state.advanced_settings['top_n_retrieval']}")
        if st.session_state.advanced_settings['enable_reranking']:
            st.write(f"**Reranking Size:** {st.session_state.advanced_settings['reranking_sample_size']}")
            st.write(f"**LLM Weight:** {st.session_state.advanced_settings['llm_weight']}")
    
    st.sidebar.markdown("---")
    
    # Pipeline steps overview
    st.sidebar.title("🗂️ RAG Pipeline Steps")
    steps = [
        "📄 PDF Upload",
        "🔄 PDF Parsing", 
        "🔧 Report Processing",
        "✂️ Text Chunking",
        "🗄️ Vector DB Creation",
        "🔍 Advanced Query & Retrieval",
        "📊 RAG System Evaluation"
    ]
    
    for i, step in enumerate(steps):
        if i <= st.session_state.pipeline_step:
            st.sidebar.success(step)
        else:
            st.sidebar.info(step)
    
    # Main interface - Document Selection
    st.subheader("📄 Document Selection")
    
    # Load available pre-made documents
    premade_docs = load_premade_documents()
    
    # Document selection options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🚀 Quick Start (Pre-made Embeddings)**")
        if premade_docs:
            premade_options = ["Select a pre-made document..."] + [
                f"{doc_info['name']} {'📊' if doc_info.get('has_evaluation') else '📖'}" 
                for doc_info in premade_docs.values()
            ]
            
            selected_premade = st.selectbox(
                "Choose from ready-to-use documents:",
                options=premade_options,
                key="premade_selector"
            )
            
            if selected_premade != "Select a pre-made document...":
                # Extract doc_id from selection - handle emoji properly
                selected_doc_name = selected_premade.split(' 📊')[0].split(' 📖')[0]  # Remove emojis
                selected_doc_id = None
                for doc_id, doc_info in premade_docs.items():
                    if doc_info['name'] == selected_doc_name:
                        selected_doc_id = doc_id
                        break
                
                if selected_doc_id:
                    st.session_state.document_type = "premade"
                    st.session_state.selected_premade = selected_doc_id
                    
                    # Show document info
                    doc_info = premade_docs[selected_doc_id]
                    st.info(f"**Selected**: {doc_info['name']}")
                    st.write(f"📝 {doc_info['description']}")
                    if doc_info.get('has_evaluation'):
                        st.success("📊 Evaluation available for this document")
                    else:
                        st.info("📖 Query-only mode (no evaluation)")
        else:
            st.warning("⚠️ No pre-made documents available.")
            with st.expander("📋 How to create pre-made embeddings"):
                st.markdown("""
                **To create pre-made embeddings:**
                
                1. **Set your OpenAI API key** in the `.env` file (rename `env` to `.env`)
                2. **Run the generation script:**
                   ```bash
                   python create_premade_embeddings.py
                   ```
                3. **Wait for processing** (5-10 minutes for both documents)
                4. **Refresh this app** to see the pre-made options
                
                **This will create:**
                - 📊 **Tradition Annual Report** (with evaluation)
                - 📖 **Harry Potter** (query-only)
                """)
                
                if st.button("🚀 Create Pre-made Embeddings Now", type="primary"):
                    if not os.getenv("OPENAI_API_KEY"):
                        st.error("❌ Please set OPENAI_API_KEY in your .env file first")
                    else:
                        st.info("Creating embeddings... This will take several minutes.")
                        try:
                            # Import and run the creation function
                            import subprocess
                            result = subprocess.run([
                                "python", "create_premade_embeddings.py"
                            ], capture_output=True, text=True, cwd=".")
                            
                            if result.returncode == 0:
                                st.success("✅ Pre-made embeddings created successfully!")
                                st.info("Please refresh the page to see the new options.")
                            else:
                                st.error(f"❌ Error creating embeddings: {result.stderr}")
                        except Exception as e:
                            st.error(f"❌ Error running creation script: {str(e)}")
    
    with col2:
        st.write("**📤 Upload Your Own Document**")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        if uploaded_file is not None:
            st.session_state.document_type = "upload"
            st.session_state.selected_premade = None
            st.info(f"**Uploaded**: {uploaded_file.name}")
            st.write("🔧 Full processing pipeline will be executed")
    
    # Reset pipeline if document type OR selected document changes
    current_document_id = f"{st.session_state.document_type}_{st.session_state.selected_premade}"
    
    if 'last_document_id' not in st.session_state:
        st.session_state.last_document_id = current_document_id
    elif st.session_state.last_document_id != current_document_id:
        # Document changed, reset pipeline completely
        keys_to_reset = ['processed_data', 'pipeline_step', 'dirs', 'sha1_name']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_document_id = current_document_id
        st.rerun()  # Force immediate rerun to refresh the interface
    
    # Handle document processing based on type
    document_ready = False
    
    if st.session_state.document_type == "premade" and st.session_state.selected_premade:
        # Pre-made document selected
        doc_info = premade_docs[st.session_state.selected_premade]
        
        if 'dirs' not in st.session_state:
            st.session_state.dirs = setup_premade_document(st.session_state.selected_premade, doc_info)
            if st.session_state.dirs:
                st.session_state.sha1_name = doc_info['sha1_name']
        
        if st.session_state.dirs:
            st.success(f"🚀 Pre-made document loaded: {doc_info['name']}")
            st.session_state.pipeline_step = 5  # Skip directly to query phase
            document_ready = True
            
        else:
            st.error("❌ Failed to setup pre-made document directories")
    
    elif st.session_state.document_type == "upload" and uploaded_file is not None:
        # Uploaded document
        if 'dirs' not in st.session_state:
            st.session_state.dirs = create_temp_directories()
            st.session_state.pdf_path = save_uploaded_pdf(uploaded_file, st.session_state.dirs['pdf_dir'])
        
        st.success(f"📄 PDF uploaded: {uploaded_file.name}")
        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 0)
        document_ready = True
    
    if document_ready:
        
        # Show different UI based on document type
        if st.session_state.document_type == "premade":
            st.info("✨ Pre-made embeddings loaded! Skip directly to querying and evaluation.")
            
        else:
            # Process button for uploaded documents
            if st.button("🚀 Start Advanced RAG Pipeline", type="primary"):
                
                # Step 1: Parse PDF
                if st.session_state.pipeline_step >= 0:
                    result1 = step1_parse_pdf(st.session_state.pdf_path, st.session_state.dirs)
                    if result1:
                        st.session_state.processed_data['step1'] = result1
                        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 1)
                    else:
                        return
                
                # Step 2: Merge reports  
                if st.session_state.pipeline_step >= 1:
                    result2 = step2_merge_reports(st.session_state.dirs, st.session_state.processed_data['step1']['sha1_name'])
                    if result2:
                        st.session_state.processed_data['step2'] = result2
                        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 2)
                    else:
                        return
                
                # Step 3: Chunk text
                if st.session_state.pipeline_step >= 2:
                    result3 = step3_chunk_text(st.session_state.dirs, st.session_state.processed_data['step1']['sha1_name'])
                    if result3:
                        st.session_state.processed_data['step3'] = result3
                        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 3)
                    else:
                        return
                
                # Step 4: Create vector database
                if st.session_state.pipeline_step >= 3:
                    result4 = step4_create_vector_db(st.session_state.dirs, st.session_state.processed_data['step1']['sha1_name'])
                    if result4:
                        st.session_state.processed_data['step4'] = result4
                        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 4)
                    else:
                        return
                
                # Step 5: Query system
                if st.session_state.pipeline_step >= 4:
                    result5 = step5_query_system(st.session_state.dirs, st.session_state.processed_data['step1']['sha1_name'], company_name="Demo Company")
                    if result5:
                        st.session_state.processed_data['step5'] = result5
                        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 5)
        
        # Show query interface if pipeline is complete
        if st.session_state.pipeline_step >= 5:
            st.markdown("---")
            
            # Get sha1_name and company name based on document type
            if st.session_state.document_type == "premade":
                sha1_name = st.session_state.sha1_name
                show_evaluation = has_evaluation_available(st.session_state.selected_premade)
                # Get the correct company name from the document info
                doc_info = premade_docs[st.session_state.selected_premade]
                company_name = doc_info['name']
            else:
                sha1_name = st.session_state.processed_data['step1']['sha1_name']
                show_evaluation = False  # No evaluation for uploaded documents (no ground truth)
                company_name = "Demo Company"
            
            # Tab interface - conditionally show evaluation
            if show_evaluation:
                tab1, tab2 = st.tabs(["🔍 Interactive Querying", "📊 System Evaluation"])
                
                with tab1:
                    step5_query_system(st.session_state.dirs, sha1_name, in_tab=True, company_name=company_name)
                
                with tab2:
                    if 'step6' in st.session_state.processed_data:
                        st.success("✅ Evaluation completed! Check results above.")
                        if st.button("🔄 Run Evaluation Again"):
                            result6 = evaluate_rag_system(st.session_state.dirs, sha1_name, company_name=company_name)
                            if result6:
                                st.session_state.processed_data['step6'] = result6
                    else:
                        evaluate_rag_system(st.session_state.dirs, sha1_name, company_name=company_name)
            else:
                # Only show query interface (no evaluation available)
                st.subheader("🔍 Interactive Querying")
                if st.session_state.document_type == "upload":
                    st.info("📝 Evaluation not available for uploaded documents (no ground truth answers)")
                step5_query_system(st.session_state.dirs, sha1_name, in_tab=True, company_name=company_name)
    
    else:
        st.info("👆 Please select a pre-made document or upload a PDF file to begin the RAG demonstration")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Pipeline"):
        # Save the API key before reset
        saved_api_key = st.session_state.openai_api_key
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.openai_api_key = saved_api_key
        st.rerun()
    
    # Information section
    st.markdown("---")
    st.markdown("""
    ### About this Advanced RAG Pipeline
    
    This demonstration showcases the complete advanced RAG (Retrieval-Augmented Generation) pipeline that won the RAG Challenge:
    
    1. **PDF Parsing**: Uses Docling to extract text, tables, and structure from PDFs
    2. **Report Processing**: Converts complex JSON to simpler page-based format
    3. **Text Chunking**: Splits documents into optimal chunks for retrieval (300 tokens, 50 overlap)
    4. **Vector Database**: Creates FAISS index with OpenAI text-embedding-3-large
    5. **Advanced Query & Retrieval**: Multi-stage retrieval with:
       - **LLM Reranking**: Improves relevance using GPT-4o-mini
       - **Parent Document Retrieval**: Gets broader page context
       - **Chain of Thought**: Structured reasoning with step-by-step analysis
    6. **System Evaluation**: Comprehensive performance assessment with:
       - **Answer Accuracy**: Comparison against ground truth answers
       - **Performance Metrics**: Response time, throughput, success rates
       - **Retrieval Quality**: Analysis of retrieved document relevance
       - **Chain-of-Thought Analysis**: Evaluation of reasoning quality
    
    **Key Features**:
    - 🔄 **LLM Reranking**: Combines vector similarity with LLM relevance scoring
    - 📜 **Parent Retrieval**: Returns full page context instead of just chunks
    - 🧠 **Chain of Thought**: Generates structured reasoning with analysis
    - ⚙️ **Advanced Settings**: Configurable parameters for optimal performance
    - 📊 **System Evaluation**: Built-in benchmarking against test questions
    - 🎯 **Performance Tracking**: Real-time accuracy and timing measurements
    
    **Evaluation Capabilities**:
    - Tests against official RAG challenge questions when available
    - Measures answer accuracy for boolean, numerical, and text questions
    - Tracks processing time, throughput, and retrieval quality
    - Provides detailed analysis of reasoning processes and errors
    - Supports comparison of different configuration settings
    
    **Requirements**: Enter your OpenAI API key in the sidebar to enable embeddings, search, and evaluation.
    """)

if __name__ == "__main__":
    main()