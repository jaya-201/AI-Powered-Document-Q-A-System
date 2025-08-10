from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os


# Step 1: Load and Split PDF
def load_pdf_chunks(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks


# Step 2: Create Embeddings using Hugging Face
def embed_chunks(chunks, path="vector_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(path)
    return db


# Step 3: Load Embedding DB or Rebuild
def load_or_rebuild_vector_db(pdf_path, embeddings, path="vector_db"):
    if os.path.exists(path):
        try:
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=False)
        except Exception as e:
            print(f"[WARNING] Failed to load vector DB safely: {e}")
    # If DB not found or failed to load, rebuild it
    print("[INFO] Rebuilding vector DB from PDF...")
    chunks = load_pdf_chunks(pdf_path)
    return embed_chunks(chunks, path)


# Step 4: Load Hugging Face LLM (Flan-T5, small)
def load_llm():
    pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    return HuggingFacePipeline(pipeline=pipe)


# Step 5: Connect LLM + Retriever
def build_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain