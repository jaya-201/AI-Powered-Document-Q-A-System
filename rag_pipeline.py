from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
import os
import pickle
from pathlib import Path
import hashlib
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# PDF_PATH= "GradCAM why did you say that.pdf"

#Load PDF
def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load() #list[Documents]

#split documents
def split_document(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

#build vectors
def build_vectorstore(splits, embeddings=None):
    if embeddings is None:
        embeddings = get_embeddings()
    return FAISS.from_documents(splits, embeddings)

def get_embeddings():
    """Singleton pattern for embeddings to avoid reloading"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=200):
    """Returns tuple: (vectorstore, document_splits)"""
    docs = load_pdf(pdf_path)
    splits = split_document(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs, splits

def save_vectorstore(vectorstore, documents=None, save_path="vector_db"):
    """Save vectorstore and documents to disk with persistence"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)
    
    # Save documents for BM25
    if documents:
        with open(f"{save_path}/documents.pkl", "wb") as f:
            pickle.dump(documents, f)
    
    print(f"Vectorstore saved to {save_path}")

def load_vectorstore(load_path="vector_db"):
    """Load vectorstore and documents from disk. Returns tuple: (vectorstore, documents)"""
    if not Path(load_path).exists():
        return None, None
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        
        # Load documents for BM25
        documents = None
        doc_path = f"{load_path}/documents.pkl"
        if Path(doc_path).exists():
            with open(doc_path, "rb") as f:
                documents = pickle.load(f)
        
        print(f"Vectorstore loaded from {load_path}")
        return vectorstore, documents
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None, None

def add_documents_to_vectorstore(vectorstore, pdf_path: str, chunk_size=1000, chunk_overlap=200):
    """Add new documents to existing vectorstore. Returns (vectorstore, new_splits)"""
    docs = load_pdf(pdf_path)
    splits = split_document(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore.add_documents(splits)
    return vectorstore, splits

def get_file_hash(file_path: str):
    """Generate hash of file for caching"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def save_document_metadata(doc_name: str, file_hash: str, metadata_path="vector_db/metadata.pkl"):
    """Save document metadata for tracking processed files"""
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    if Path(metadata_path).exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    metadata[doc_name] = file_hash
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def load_document_metadata(metadata_path="vector_db/metadata.pkl"):
    """Load document metadata"""
    if not Path(metadata_path).exists():
        return {}
    try:
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def is_document_processed(doc_name: str, file_hash: str, metadata_path="vector_db/metadata.pkl"):
    """Check if document is already processed"""
    metadata = load_document_metadata(metadata_path)
    return metadata.get(doc_name) == file_hash

def merge_documents(existing_docs, new_docs):
    """Merge new documents with existing ones"""
    if existing_docs is None:
        return new_docs
    if new_docs is None:
        return existing_docs
    return existing_docs + new_docs

def hybrid_retrieve(dense_retriever, sparse_retriever, query: str, k: int = 3) -> List[Document]:
    """
    Manual ensemble retrieval combining dense and sparse retrievers
    """
    # Get results from both retrievers
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)
    
    # Merge and deduplicate based on content
    seen_content = set()
    merged_docs = []
    
    # Add documents from both retrievers, alternating for diversity
    all_docs = []
    for i in range(max(len(dense_docs), len(sparse_docs))):
        if i < len(dense_docs):
            all_docs.append(dense_docs[i])
        if i < len(sparse_docs):
            all_docs.append(sparse_docs[i])
    
    # Deduplicate and limit to k results
    for doc in all_docs:
        if doc.page_content not in seen_content and len(merged_docs) < k:
            seen_content.add(doc.page_content)
            merged_docs.append(doc)
    
    return merged_docs

# Gemini API setup
llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-1.5-flash-latest"),
    temperature=0.3,  # Slightly higher for more natural responses
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent document assistant. Your role is to answer questions based on the provided context.\n\n"
     "Rules:\n"
     "1. Read and understand the context carefully\n"
     "2. Provide clear, natural, conversational answers\n"
     "3. Synthesize information from the context - don't just copy-paste\n"
     "4. If the context doesn't contain enough information to answer, say 'I don't have enough information to answer that'\n"
     "5. Be helpful and concise\n"
     "6. You can rephrase and explain concepts from the context in your own words\n"
     "7. Never make up information not present in the context"),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a clear, conversational answer based on the context:")
])

def format_docs(docs):
    if not docs:
        return "No relevant information found."
    # Just join the content without chunk labels for cleaner context
    return "\n\n".join(doc.page_content for doc in docs)

#top level run with hybrid retrieval
def query_with_vectorstore(vectorstore, question: str, documents=None, k: int = 3):
    """
    Hybrid retrieval: Combines dense (FAISS semantic) + sparse (BM25 keyword) search
    """
    # Dense retriever (semantic search with FAISS)
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": 0.5
        }
    )
    
    # If documents available, use hybrid retrieval
    if documents:
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = k
        
        # Get hybrid results using our custom function
        retrieved_docs = hybrid_retrieve(dense_retriever, sparse_retriever, question, k)
        
        # Format and use in chain
        context = format_docs(retrieved_docs)
    else:
        # Fallback to dense-only if no documents
        retrieved_docs = dense_retriever.invoke(question)
        context = format_docs(retrieved_docs)
    
    # Create response using LLM
    formatted_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(formatted_prompt)
    
    return answer.content

def query_with_chat_history(vectorstore, question: str, chat_history: list, documents=None, k: int = 3):
    """
    Query with conversation history for continuous chat
    """
    # Dense retriever (semantic search with FAISS)
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": 0.5
        }
    )
    
    # If documents available, use hybrid retrieval
    if documents:
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = k
        
        # Get hybrid results using our custom function
        retrieved_docs = hybrid_retrieve(dense_retriever, sparse_retriever, question, k)
        
        # Format and use in chain
        context = format_docs(retrieved_docs)
    else:
        # Fallback to dense-only if no documents
        retrieved_docs = dense_retriever.invoke(question)
        context = format_docs(retrieved_docs)
    
    # Format chat history for context
    history_text = ""
    if chat_history:
        history_text = "\\n\\nPrevious conversation:\\n"
        for i, exchange in enumerate(chat_history[-5:], 1):  # Last 5 exchanges
            history_text += f"Q{i}: {exchange['question']}\\n"
            history_text += f"A{i}: {exchange['answer']}\\n"
    
    # Create prompt with history
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent document assistant engaged in a conversation. "
         "Use the provided context and previous conversation to answer questions naturally.\\n\\n"
         "Rules:\\n"
         "1. Remember the conversation history and refer to it when relevant\\n"
         "2. If the user asks 'it', 'that', 'this', etc., use context from previous questions\\n"
         "3. Provide clear, natural, conversational answers\\n"
         "4. Synthesize information from both the document context and conversation\\n"
         "5. If you don't have enough information, say so\\n"
         "6. Be helpful and maintain conversation flow"),
        ("human", "Document Context:\\n{context}{history}\\n\\nCurrent Question: {question}\\n\\nAnswer:")
    ])
    
    formatted_prompt = chat_prompt.format(
        context=context,
        history=history_text,
        question=question
    )
    answer = llm.invoke(formatted_prompt)
    
    return answer.content

# if __name__ == "__main__":
#     print("PDF RAG ready (HuggingFace mode). Ask a question (or Ctrl+C to exit).")
#     q = input("\nQ: ").strip()
#     ans = query_with_vectorstore(PDF_PATH, q)
#     print("\nA:", ans)