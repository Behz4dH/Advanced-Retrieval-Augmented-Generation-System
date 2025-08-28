#!/usr/bin/env python3
"""
Script to create pre-made embeddings for demo documents.
This ensures users have instant access to working examples.
"""

import os
import json
import shutil
from pathlib import Path
from src.pipeline import Pipeline, RunConfig

def create_premade_embeddings():
    """Create embeddings for demo documents"""
    
    # Configuration for embedding creation
    config = RunConfig(
        use_serialized_tables=False,
        parent_document_retrieval=True,
        llm_reranking=True,
        parallel_requests=5,
        config_suffix="_premade"
    )
    
    premade_dir = Path("data/premade_embeddings")
    premade_dir.mkdir(exist_ok=True)
    
    # Document definitions
    documents = {
        "test_tradition": {
            "name": "Tradition Annual Report",
            "pdf_path": Path("data/test_set/pdf_reports/2779336b845a41544348abb7b3e6e5bd2ff893a2.pdf"),
            "has_evaluation": True,
            "description": "Financial annual report with numerical data and metrics"
        },
        "harrypotter": {
            "name": "Harry Potter",
            "pdf_path": Path("data/harrypotter.pdf"),
            "has_evaluation": False,
            "description": "Literary content for demonstrating text-based queries"
        }
    }
    
    print("Creating pre-made embeddings...")
    
    for doc_id, doc_info in documents.items():
        print(f"\nProcessing {doc_info['name']}...")
        
        # Create document directory
        doc_dir = premade_dir / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        pdf_dir = doc_dir / "pdf_reports"
        pdf_dir.mkdir(exist_ok=True)
        
        # Copy PDF to document directory
        pdf_dest = pdf_dir / doc_info['pdf_path'].name
        if not pdf_dest.exists():
            shutil.copy2(doc_info['pdf_path'], pdf_dest)
            print(f"   Copied PDF: {doc_info['pdf_path'].name}")
        
        # Create CSV metadata
        csv_path = doc_dir / "subset.csv"
        if not csv_path.exists():
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(f"sha1,company_name\n")
                f.write(f"{doc_info['pdf_path'].stem},{doc_info['name']}\n")
            print(f"   Created metadata CSV")
        
        # Initialize pipeline for this document
        pipeline = Pipeline(doc_dir, run_config=config)
        
        try:
            # Step 1: Parse PDF
            print(f"   Parsing PDF...")
            pipeline.parse_pdf_reports(parallel=False)  # Use sequential for reliability
            
            # Step 2: Process reports
            print(f"   Processing reports...")
            pipeline.merge_reports()
            
            # Step 3: Chunk text
            print(f"   Chunking text...")
            pipeline.chunk_reports()
            
            # Step 4: Create embeddings
            print(f"   Creating vector database...")
            pipeline.create_vector_dbs()
            
            # Create document info file
            info_path = doc_dir / "document_info.json"
            doc_info_extended = {
                **doc_info,
                "pdf_path": str(doc_info['pdf_path']),
                "doc_id": doc_id,
                "sha1_name": doc_info['pdf_path'].stem,
                "processing_completed": True
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(doc_info_extended, f, indent=2, ensure_ascii=False)
            
            print(f"   SUCCESS: {doc_info['name']} embeddings created successfully!")
            
        except Exception as e:
            print(f"   ERROR processing {doc_info['name']}: {str(e)}")
            continue
    
    print(f"\nPre-made embeddings creation completed!")
    print(f"Embeddings saved to: {premade_dir.absolute()}")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("src/pipeline.py").exists():
        print("ERROR: Please run this script from the project root directory")
        exit(1)
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("   You can set it in the .env file or as an environment variable")
        exit(1)
    
    create_premade_embeddings()