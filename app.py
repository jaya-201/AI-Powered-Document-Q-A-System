import streamlit as st
import os
from rag_pipeline import load_pdf_chunks, embed_chunks, load_or_rebuild_vector_db, build_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("AI-Powered Document Q&A")

if "db_created" not in st.session_state:
    st.session_state.db_created = False

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing PDF...")
    chunks = load_pdf_chunks("uploaded.pdf")
    embed_chunks(chunks)
    st.session_state.db_created = True
    st.success("Document indexed and ready!")

if st.session_state.db_created:
    db = load_or_rebuild_vector_db("uploaded.pdf", embeddings)
    qa_chain = build_qa_chain(db)

    query = st.text_input("Ask a question about the document:")
    if query:
        result = qa_chain({"query": query})
        if not result.get("source_documents") or len(result["source_documents"]) == 0:
            st.warning("No result found, ask question related to the PDF.")
        else:
            st.subheader("Answer:")
            st.write(result["result"])

        with st.expander("Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)