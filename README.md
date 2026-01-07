# RAG Document Q&A Application

A full-stack document Q&A application using Retrieval-Augmented Generation (RAG) with FastAPI backend and React frontend.

## Features

- 🚀 FastAPI backend with REST API
- ⚛️ React frontend with modern UI
- 📚 Multi-document support
- 💾 Persistent vectorstore (FAISS)
- 🔍 Smart caching (no duplicate processing)
- 🤖 HuggingFace LLM (Flan-T5)
- 📄 PDF document processing

## Project Structure

```
.
├── main.py                 # FastAPI server
├── rag_pipeline.py         # RAG logic
├── requirements.txt        # Python dependencies
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   ├── App.css        # Styles
│   │   └── index.js       # Entry point
│   └── package.json       # Node dependencies
├── vector_db/             # Persisted vectorstore
└── uploads/               # Uploaded PDFs
```

## Setup & Installation

### Backend Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the FastAPI server:**
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will run on: `http://localhost:8000`

API docs available at: `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install Node dependencies:**
```bash
npm install
```

3. **Start React development server:**
```bash
npm start
```

Frontend will run on: `http://localhost:3000`

## API Endpoints

### GET `/`
Health check endpoint

### GET `/status`
Get system status and document count

### GET `/documents`
List all processed documents

### POST `/upload`
Upload and process a PDF document
- **Body:** multipart/form-data with `file` field

### POST `/query`
Query the vectorstore
- **Body:** 
```json
{
  "question": "Your question here",
  "k": 3
}
```

### DELETE `/documents`
Clear all documents and reset vectorstore

## Usage

1. Start the backend server (port 8000)
2. Start the frontend (port 3000)
3. Open browser to `http://localhost:3000`
4. Upload PDF documents
5. Ask questions about your documents

## Technologies

**Backend:**
- FastAPI
- LangChain
- FAISS (vector database)
- HuggingFace Transformers
- Sentence Transformers

**Frontend:**
- React 18
- Axios (HTTP client)
- Modern CSS with gradients

## Advanced Features

- ✅ Vectorstore persistence
- ✅ Document caching (MD5 hashing)
- ✅ Multi-document support
- ✅ RESTful API architecture
- ✅ CORS enabled
- ✅ Error handling
- ✅ Loading states

## Future Enhancements

- [ ] Multi-query retrieval
- [ ] Hybrid search (BM25 + Semantic)
- [ ] Document compression
- [ ] Re-ranking with cross-encoders
- [ ] Source citations
- [ ] User authentication
- [ ] Chat history
- [ ] Metadata filtering
