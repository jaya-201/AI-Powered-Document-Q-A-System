from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
from typing import List, Optional
from contextlib import asynccontextmanager

from rag_pipeline import (
    setup_pipeline,
    query_with_vectorstore,
    query_with_chat_history,
    load_vectorstore,
    save_vectorstore,
    add_documents_to_vectorstore,
    get_file_hash,
    is_document_processed,
    save_document_metadata,
    load_document_metadata,
    merge_documents
)

# Global vectorstore and documents
vectorstore = None
documents = None  # For BM25 retrieval
chat_sessions = {}  # Store conversation history by session_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vectorstore on startup"""
    global vectorstore, documents
    vectorstore, documents = load_vectorstore()
    if vectorstore:
        print("✅ Loaded existing vectorstore")
    else:
        print("ℹ️ No existing vectorstore found")
    yield
    # Cleanup on shutdown if needed

app = FastAPI(title="RAG Document Q&A API", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    k: int = 3
    session_id: Optional[str] = None  # For chat history tracking

class QueryResponse(BaseModel):
    answer: str
    question: str

class DocumentInfo(BaseModel):
    name: str
    processed: bool

class StatusResponse(BaseModel):
    status: str
    documents_count: int
    documents: List[str]

@app.get("/")
async def root():
    return {"message": "RAG Document Q&A API", "status": "running"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status and processed documents"""
    metadata = load_document_metadata()
    return StatusResponse(
        status="ready" if vectorstore else "no_documents",
        documents_count=len(metadata),
        documents=list(metadata.keys())
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    global vectorstore, documents
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create uploads directory
    Path("uploads").mkdir(exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Check if already processed
    file_hash = get_file_hash(file_path)
    
    if is_document_processed(file.filename, file_hash):
        return {
            "message": f"Document '{file.filename}' already processed",
            "status": "skipped",
            "filename": file.filename
        }
    
    # Process document
    try:
        if vectorstore is None:
            # Create new vectorstore
            vectorstore, new_docs = setup_pipeline(file_path)
            documents = new_docs
            message = f"Created knowledge base with '{file.filename}'"
        else:
            # Add to existing vectorstore
            vectorstore, new_docs = add_documents_to_vectorstore(vectorstore, file_path)
            documents = merge_documents(documents, new_docs)
            message = f"Added '{file.filename}' to knowledge base"
        
        # Save vectorstore and metadata
        save_vectorstore(vectorstore, documents)
        save_document_metadata(file.filename, file_hash)
        
        return {
            "message": message,
            "status": "success",
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the vectorstore with hybrid retrieval and chat history"""
    global vectorstore, documents, chat_sessions
    
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get or create session history
        session_id = request.session_id or "default"
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_history = chat_sessions[session_id]
        
        # Query with chat history
        answer = query_with_chat_history(
            vectorstore=vectorstore,
            question=request.question,
            chat_history=chat_history,
            documents=documents,
            k=request.k
        )
        
        # Update chat history
        chat_sessions[session_id].append({
            "question": request.question,
            "answer": answer
        })
        
        # Keep only last 10 exchanges to avoid token limits
        if len(chat_sessions[session_id]) > 10:
            chat_sessions[session_id] = chat_sessions[session_id][-10:]
        
        return QueryResponse(answer=answer, question=request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/documents", response_model=List[str])
async def list_documents():
    """Get list of all processed documents"""
    metadata = load_document_metadata()
    return list(metadata.keys())

@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents and reset vectorstore"""
    global vectorstore, documents
    
    try:
        if Path("vector_db").exists():
            shutil.rmtree("vector_db")
        
        vectorstore = None
        documents = None
        
        return {
            "message": "All documents cleared successfully",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
