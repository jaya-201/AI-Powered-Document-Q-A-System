AI-Powered Document Q&A System

This is a Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask natural language questions about its content.  
It uses LangChain, FAISS, Hugging Face Embeddings, and Flan-T5(text-to-text transformer model) for the LLM, wrapped in a Streamlit UI.

Features:
- Upload a PDF and automatically index it into vector embeddings.
- Store embeddings locally using FAISS.
- Ask questions and retrieve precise answers from the document.
- Displays the text chunks used to generate each answer.

How It Works
1. Load PDF: 
   The PDF is loaded using 'PyMuPDFLoader' from LangChain.
2. Chunk Splitting:
   Text is split into overlapping chunks with 'RecursiveCharacterTextSplitter'.
3. Embeddings:
   Uses 'sentence-transformers/all-MiniLM-L6-v2' to generate vector embeddings.
4. Vector Storage:
   FAISS(Facebook AI Similarity Search) stores the embeddings locally for fast retrieval.
5. LLM:
   Uses 'google/flan-t5-base' for question answering.
6. RetrievalQA Chain:
   Combines retriever + LLM to return answers along with the source chunks.

Installation
1. Clone this repository 
   bash:
   git clone https://github.com/your-username/pdf-rag-app.git
   cd pdf-rag-app
2. Create a virtual environment (recommended)
   bash:
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
3. Install dependencies
   bash:
   pip install -r requirements.txt
4. Usage
   bash:
   streamlit run app.py
