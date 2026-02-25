import streamlit as st
import os
import httpx
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from pdf_chunker import PDFChunker
from openai_embeddings import OpenAIEmbeddingClient


# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment")
    st.stop()

# Shared corporate-safe HTTP client
http_client = httpx.Client(verify=False)

# OpenAI SDK client
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=http_client,
)

# Custom embeddings adapter
embeddings = OpenAIEmbeddingClient(
    api_key=OPENAI_API_KEY,
    http_client=http_client,
)

# --------------------------------------------------
# Page Setup
# --------------------------------------------------

st.set_page_config(page_title="Simple FAISS RAG", layout="wide")

st.title("📦 Simple Vector RAG (FAISS)")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["text-embedding-3-small", "text-embedding-3-large"],
)

answer_model = st.sidebar.selectbox(
    "Answer Model",
    ["gpt-4.1-mini", "gpt-4.1", "gpt-5", "gpt-5.2", "gpt-5.2-mini"],
)

chunk_size_k = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 3)
persist_path = st.sidebar.text_input("FAISS Path", "faiss_store")

pdf_dir = st.text_input("PDF Directory", r"D:\DesktopData\KG\data")

# --------------------------------------------------
# Ingestion
# --------------------------------------------------

st.header("📥 Ingestion")

if st.button("Build FAISS Index"):

    if not os.path.isdir(pdf_dir):
        st.error("Invalid directory")
    else:
        chunker = PDFChunker()
        docs = []

        for file in os.listdir(pdf_dir):
            if file.lower().endswith(".pdf"):
                chunks = chunker.extract(os.path.join(pdf_dir, file), chunk_size_k)
                docs.extend(chunks)

        if docs:
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(persist_path)
            st.success("Index built!")
        else:
            st.error("No PDFs found.")

# --------------------------------------------------
# Retrieval
# --------------------------------------------------

st.header("🔎 Retrieval")

question = st.text_area("Ask a question")

if st.button("Get Answer"):

    if not os.path.exists(persist_path):
        st.error("Index missing")

    elif not question.strip():
        st.warning("Enter a question")

    else:
        vectorstore = FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        docs = vectorstore.similarity_search(question, k=top_k)
        context = "\n\n".join(d.page_content for d in docs)

        llm = ChatOpenAI(
            model=answer_model,
            api_key=OPENAI_API_KEY,
            http_client=http_client,
            temperature=0,
        )

        response = llm.invoke(f"Context:\n{context}\n\nQuestion:\n{question}")

        st.subheader("📌 Answer")
        st.write(response.content)
