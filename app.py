import streamlit as st
from rag_modified import setup_pipeline, query_with_vectorstore

st.title("AI-Powered Document Q&A")

# Store vectorstore in session_state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing PDF...")
    st.session_state.vectorstore = setup_pipeline("uploaded.pdf")
    st.success("Document indexed and ready!")

if st.session_state.vectorstore is not None:
    query = st.text_input("Ask a question about the document:")
    if query:
        result = query_with_vectorstore(st.session_state.vectorstore, query, k=3)

        st.subheader("Answer:")
        st.write(result)