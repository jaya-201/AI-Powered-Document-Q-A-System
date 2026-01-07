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
    load_vectorstore,
    save_vectorstore,
    add_documents_to_vectorstore,
    get_file_hash,
    is_document_processed,
    save_document_metadata,
    load_document_metadata
)

# Global vectorstore
vectorstore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vectorstore on startup"""
    global vectorstore
    vectorstore = load_vectorstore()
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
    global vectorstore
    
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
            vectorstore = setup_pipeline(file_path)
            message = f"Created knowledge base with '{file.filename}'"
        else:
            # Add to existing vectorstore
            vectorstore = add_documents_to_vectorstore(vectorstore, file_path)
            message = f"Added '{file.filename}' to knowledge base"
        
        # Save vectorstore and metadata
        save_vectorstore(vectorstore)
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
    """Query the vectorstore with a question"""
    global vectorstore
    
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = query_with_vectorstore(vectorstore, request.question, k=request.k)
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
    global vectorstore
    
    try:
        if Path("vector_db").exists():
            shutil.rmtree("vector_db")
        
        vectorstore = None
        
        return {
            "message": "All documents cleared successfully",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
