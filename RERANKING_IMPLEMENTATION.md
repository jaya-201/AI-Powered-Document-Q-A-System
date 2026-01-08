# Advanced RAG Implementation - Cross-Encoder Reranking

## Overview
Successfully integrated cross-encoder reranking for multi-step retrieval in the RAG pipeline.

## What Was Implemented

### 1. Cross-Encoder Reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` from HuggingFace
- **Purpose**: Rerank retrieved documents by computing query-document relevance scores
- **Advantage**: More accurate than bi-encoder similarity (FAISS), but slower - ideal as second stage

### 2. Multi-Step Retrieval Pipeline

The complete retrieval now works in 4 stages:

```
Query → [1] Dense Retrieval (FAISS) 
      ↓
      [2] Sparse Retrieval (BM25)
      ↓  
      [3] Hybrid Merge (deduplicate)
      ↓
      [4] Cross-Encoder Reranking (score & reorder)
      ↓
      Final Top-K Documents → LLM
```

### 3. Implementation Details

#### Lazy Loading Pattern
All heavy models (LLM, Cross-Encoder) are lazy-loaded to avoid startup delays:

```python
# Cross-encoder loaded only when first needed
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder
```

#### Reranking Function
```python
def rerank_documents(documents, query, top_k=3):
    """
    Rerank documents using cross-encoder for better relevance.
    Returns top_k most relevant documents.
    """
    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    # Sort by scores and return top_k
```

#### Updated Query Functions
Both `query_with_vectorstore()` and `query_with_chat_history()` now use reranking:

```python
# Retrieve more candidates for reranking
hybrid_docs = hybrid_retrieve(dense_retriever, sparse_retriever, question, k * 2)

# Rerank to get best k documents
retrieved_docs = rerank_documents(hybrid_docs, question, top_k=k)
```

## Files Modified

### 1. rag_pipeline.py
- Added lazy-loaded cross-encoder initialization
- Added `rerank_documents()` function
- Updated `query_with_vectorstore()` to use reranking
- Updated `query_with_chat_history()` to use reranking
- Made LLM initialization lazy as well

### 2. New File: download_reranker.py
- Script to download the cross-encoder model
- Caches model locally for faster subsequent loads
- Includes test to verify model works

## How to Use

### First Time Setup
Run this once to download and cache the reranker model:

```powershell
cd "c:\Users\Jaya\OneDrive\Documents\SelfProjects\LangChain Project"
.\venv\Scripts\Activate.ps1
python download_reranker.py
```

This will download the ~80MB model and cache it locally.

### Start the Backend
```powershell
python main.py
```

The backend will start immediately. The cross-encoder loads only when you first query a document.

### Expected Behavior
1. **First query**: Cross-encoder model loads (~2-3 seconds), then shows reranking scores
2. **Subsequent queries**: Instant - model is already loaded

Example output:
```
Loading cross-encoder model for reranking...
Cross-encoder loaded successfully
Reranked 6 documents, returning top 3
Top scores: ['8.2341', '6.1234', '5.0923']
```

## Benefits of Reranking

### 1. Better Relevance
- Cross-encoders jointly encode query+document, capturing more nuanced relationships
- More accurate than dot product similarity used by FAISS

### 2. Improved Answer Quality
- LLM receives the most relevant context
- Reduces hallucinations by filtering out less relevant chunks

### 3. Efficient Two-Stage Design
- Fast hybrid retrieval gets many candidates (cheap)
- Expensive cross-encoder only ranks top candidates
- Best balance of speed and accuracy

## Configuration

### Adjust Number of Candidates
In query functions, change the multiplier:

```python
# Retrieve 2x candidates for reranking
hybrid_docs = hybrid_retrieve(..., k * 2)

# Or retrieve 3x for more thorough reranking
hybrid_docs = hybrid_retrieve(..., k * 3)
```

### Adjust Final Results
Change the `k` parameter when calling:

```python
answer = query_with_chat_history(vectorstore, question, chat_history, documents, k=5)
```

## Advanced RAG Features Summary

Your RAG system now includes:

✅ **Multi-Document Support** - Upload and query multiple PDFs
✅ **Persistent Vectorstore** - FAISS index persists between restarts  
✅ **Hybrid Retrieval** - Dense (FAISS) + Sparse (BM25) search
✅ **Cross-Encoder Reranking** - Multi-step retrieval with relevance scoring
✅ **Chat History** - Conversational memory with session management
✅ **Gemini API Integration** - Cloud LLM with fast responses
✅ **FastAPI Backend** - RESTful API with CORS support
✅ **React Frontend** - Modern chat interface

## Troubleshooting

### If Model Download Fails
- Check internet connection
- Run `download_reranker.py` separately to diagnose
- Model will download to: `~/.cache/huggingface/hub/`

### If Startup is Slow
- The first query will load the cross-encoder (one-time delay)
- After that, queries are fast
- Lazy loading prevents startup issues

### If Out of Memory
- Reduce candidates: Use `k * 1.5` instead of `k * 2`
- Reduce final results: Use `k=2` instead of `k=3`

## Next Steps (Optional Enhancements)

1. **Query Expansion**: Generate multiple query variations before retrieval
2. **Metadata Filtering**: Filter by document name, page number, etc.
3. **Feedback Loop**: Let users rate answers to fine-tune relevance
4. **Streaming Responses**: Stream LLM output for faster perceived response
5. **Caching**: Cache query-answer pairs for identical questions

## Technical Stack

- **Embedding**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Dense Retrieval**: FAISS 1.13.2
- **Sparse Retrieval**: BM25 (rank-bm25 0.2.2)
- **Reranker**: Cross-Encoder ms-marco-MiniLM-L-6-v2
- **LLM**: Google Gemini 2.5 Flash via API
- **Backend**: FastAPI 0.128.0
- **Frontend**: React 18
