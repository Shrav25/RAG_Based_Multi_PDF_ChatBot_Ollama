import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load and split all PDFs
@st.cache_resource
def load_and_index_docs(folder_path="my_docs/"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="rag_chroma"
    )
    return vectordb.as_retriever()

# Initialize retriever and QA chain
retriever = load_and_index_docs()
llm = Ollama(model="gemma3:4b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Streamlit UI
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📚 PDF Chatbot (RAG with Ollama)")

query = st.text_input("Ask a question based on the uploaded PDFs:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.success(result["result"])
