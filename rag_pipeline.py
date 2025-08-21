from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os

# PDF_PATH= "GradCAM why did you say that.pdf"

#Load PDF
def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load() #list[Documents]

#split documents
def split_document(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

#build vectors
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

def setup_pipeline(pdf_path: str, chunk_size=500, chunk_overlap=50):
    docs = load_pdf(pdf_path)
    splits = split_document(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

#Hugging Face LLM setup
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_new_tokens=512
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict document Q&A bot. Use ONLY the provided context. "
     "If the context is empty or unrelated, answer exactly with: 'I don't know'."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

#top level run
def query_with_vectorstore(vectorstore, question: str, k: int = 3):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

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