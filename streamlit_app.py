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

def step5_query_system(dirs: Dict[Path, Path], sha1_name: str) -> Dict:
    """Step 5: Advanced Query & Retrieval with CoT"""
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
            query = st.text_input("Enter your question:", placeholder="What would you like to know about this document?")
        with col2:
            st.write("**Sample queries:**")
            for sample in sample_queries:
                if st.button(sample, key=f"sample_{sample[:20]}"):
                    query = sample
        
        if query and st.button("🔍 Advanced Search", type="primary"):
            with st.spinner("Performing advanced search..."):
                try:
                    # Performance tracking
                    start_time = time.time()
                    retrieval_start = time.time()
                    
                    # Perform retrieval based on settings
                    if st.session_state.advanced_settings['enable_reranking'] and isinstance(retriever, HybridRetriever):
                        results = retriever.retrieve_and_rerank(
                            company_name="Demo Company",
                            query=query,
                            top_n=st.session_state.advanced_settings['top_n_retrieval'],
                            llm_reranking_sample_size=st.session_state.advanced_settings['reranking_sample_size'],
                            return_parent_pages=st.session_state.advanced_settings['enable_parent_retrieval'],
                            llm_weight=st.session_state.advanced_settings['llm_weight']
                        )
                    else:
                        results = retriever.retrieve_by_company_name(
                            company_name="Demo Company",
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

def main():
    st.title("🔍 RAG Pipeline Demo")
    st.markdown("**Upload a PDF and watch the complete RAG process in action!**")
    
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
        "🔍 Advanced Query & Retrieval"
    ]
    
    for i, step in enumerate(steps):
        if i <= st.session_state.pipeline_step:
            st.sidebar.success(step)
        else:
            st.sidebar.info(step)
    
    # Main interface
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Step 0: Setup
        if 'dirs' not in st.session_state:
            st.session_state.dirs = create_temp_directories()
            st.session_state.pdf_path = save_uploaded_pdf(uploaded_file, st.session_state.dirs['pdf_dir'])
        
        st.success(f"📄 PDF uploaded: {uploaded_file.name}")
        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 0)
        
        # Process button  
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
                result5 = step5_query_system(st.session_state.dirs, st.session_state.processed_data['step1']['sha1_name'])
                if result5:
                    st.session_state.processed_data['step5'] = result5
                    st.session_state.pipeline_step = max(st.session_state.pipeline_step, 5)
        
        # Show query interface if pipeline is complete
        if st.session_state.pipeline_step >= 5 and 'step5' in st.session_state.processed_data:
            st.markdown("---")
            step5_query_system(st.session_state.dirs, st.session_state.processed_data['step1']['sha1_name'])
    
    else:
        st.info("👆 Please upload a PDF file to begin the RAG demonstration")
    
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
       - **Performance Metrics**: Detailed timing and accuracy stats
    
    **Key Features**:
    - 🔄 **LLM Reranking**: Combines vector similarity with LLM relevance scoring
    - 📜 **Parent Retrieval**: Returns full page context instead of just chunks
    - 🧠 **Chain of Thought**: Generates structured reasoning with analysis
    - ⚙️ **Advanced Settings**: Configurable parameters for optimal performance
    - 📊 **Performance Metrics**: Real-time timing and accuracy measurements
    
    **Requirements**: Enter your OpenAI API key in the sidebar to enable embeddings and advanced search.
    """)

if __name__ == "__main__":
    main()