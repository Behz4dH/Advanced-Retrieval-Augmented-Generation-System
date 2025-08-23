# Running the RAG Pipeline Demo

This Streamlit app demonstrates the complete RAG pipeline from the challenge-winning solution.

## Prerequisites

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **OpenAI API Key:**
   You can either:
   - Enter your API key directly in the Streamlit app sidebar (recommended)
   - Or set it as an environment variable in `.env` file:
   ```bash
   # Create .env file (copy from env template)
   cp env .env
   
   # Add your OpenAI API key to .env file:
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the App

```bash
streamlit run streamlit_app.py
```

## What the Demo Shows

The app will take you through each step of the RAG pipeline:

1. **📄 PDF Upload** - Upload any PDF document
2. **🔄 PDF Parsing** - Uses Docling to extract text, tables, and structure
3. **🔧 Report Processing** - Converts to simpler page-based format
4. **✂️ Text Chunking** - Splits into optimal chunks for retrieval
5. **🗄️ Vector Database** - Creates FAISS index with OpenAI embeddings
6. **🔍 Query & Retrieval** - Search through the document using vector similarity

## Features

- **Real-time Processing**: Watch each pipeline step execute
- **Interactive Querying**: Ask questions about your uploaded document
- **Detailed Metrics**: See processing times, chunk counts, similarity scores
- **Step Visualization**: Progress tracker showing completed steps

## Sample Queries

Try asking questions like:
- "What is the main topic of this document?"
- "What are the key findings mentioned?"
- "Tell me about any financial information"
- "What recommendations are provided?"

## Technical Details

- Uses the same components as the challenge-winning solution
- OpenAI's text-embedding-3-large for embeddings (3072 dimensions)
- FAISS for efficient vector similarity search
- Supports all PDF types that Docling can parse

## Troubleshooting

- **Missing API Key**: Enter your OpenAI API key in the sidebar (🔑 field)
- **Parsing Errors**: Some complex PDFs may fail - try a simpler document
- **Memory Issues**: Large documents may require more RAM for processing

## Reset

Use the "🔄 Reset Pipeline" button in the sidebar to start over with a new document.