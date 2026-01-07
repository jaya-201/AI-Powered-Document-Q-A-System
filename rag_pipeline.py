from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pickle
from pathlib import Path
import hashlib
from dotenv import load_dotenv

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
    docs = load_pdf(pdf_path)
    splits = split_document(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

def save_vectorstore(vectorstore, save_path="vector_db"):
    """Save vectorstore to disk with persistence"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Vectorstore saved to {save_path}")

def load_vectorstore(load_path="vector_db"):
    """Load vectorstore from disk"""
    if not Path(load_path).exists():
        return None
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from {load_path}")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def add_documents_to_vectorstore(vectorstore, pdf_path: str, chunk_size=1000, chunk_overlap=200):
    """Add new documents to existing vectorstore"""
    docs = load_pdf(pdf_path)
    splits = split_document(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore.add_documents(splits)
    return vectorstore

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

# Gemini API setup
llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-1.5-flash-latest"),
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise document Q&A assistant. Follow these rules strictly:\n"
     "1. Use ONLY information from the provided context\n"
     "2. If the answer is not in the context, respond: 'I don't know'\n"
     "3. Do not use external knowledge or make assumptions\n"
     "4. Quote relevant parts of the context when answering\n"
     "5. Keep answers concise and factual"),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above:")
])

def format_docs(docs):
    if not docs:
        return "No relevant information found."
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Chunk {i}]\n{doc.page_content}")
    return "\n\n".join(formatted)

#top level run
def query_with_vectorstore(vectorstore, question: str, k: int = 3):
    # Use MMR (Maximum Marginal Relevance) for diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,  # Fetch more candidates
            "lambda_mult": 0.5  # Balance relevance vs diversity
        }
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()
    return chain.invoke(question)

# if __name__ == "__main__":
#     print("PDF RAG ready (HuggingFace mode). Ask a question (or Ctrl+C to exit).")
#     q = input("\nQ: ").strip()
#     ans = query_with_vectorstore(PDF_PATH, q)
#     print("\nA:", ans)