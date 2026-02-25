import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

from kg_rag_engine import KGRAGEngine

# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# Asset paths
# --------------------------------------------------

ASSETS = Path(__file__).parent / "assets"
IMPETUS_LOGO = ASSETS / "impetus_logo.png"
NEO4J_LOGO = ASSETS / "neo4j_logo.png"

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="InsightGraph AI",
    layout="wide",
)

# --------------------------------------------------
# Header Branding
# --------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if IMPETUS_LOGO.exists():
        st.image(str(IMPETUS_LOGO), width=180)

# st.title("InsightGraph AI — Query. Discover. Understand")

st.markdown(
    """
    <h2 style='text-align: center; margin-bottom: 0.2em;'>
    InsightGraph AI — Query. Discover. Understand
    </h2>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "ℹ️ Hover for platform description",
    help="""
Knowledge Graph powered RAG system combining:
• vector similarity
• graph traversal
• multi-hop reasoning
• LLM answer synthesis
""",
)

# --------------------------------------------------
# Sidebar – Neo4j Logo + Configuration
# --------------------------------------------------

if NEO4J_LOGO.exists():
    col1, col2, col3 = st.sidebar.columns([1, 3, 1])
    with col2:
        st.image(str(NEO4J_LOGO), width=150)

st.sidebar.markdown("### Neo4j Configuration")

neo4j_uri = st.sidebar.text_input(
    "Neo4j URI",
    value=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
)

neo4j_user = st.sidebar.text_input(
    "Neo4j Username",
    value=os.getenv("NEO4J_USER", "neo4j"),
)

neo4j_password = st.sidebar.text_input(
    "Neo4j Password",
    type="password",
    value=os.getenv("NEO4J_PASSWORD", ""),
)

database = st.sidebar.text_input("Database", value="insurancedetail")

vector_index = st.sidebar.text_input("Vector Index", value="graph_store")

# --------------------------------------------------
# Sidebar – Retrieval Settings
# --------------------------------------------------

st.sidebar.markdown("### Retrieval Settings")

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["text-embedding-3-small", "text-embedding-3-large"],
)

answer_model = st.sidebar.selectbox(
    "Answer Model",
    [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5",
        "gpt-5.2",
        "gpt-5.2-mini",
    ],
)

top_k = st.sidebar.slider("Top-K Vector Seeds", 1, 10, 3)

similarity_threshold = st.sidebar.slider(
    "Graph Similarity Threshold",
    0.5,
    0.9,
    0.75,
    step=0.05,
)

# --------------------------------------------------
# Question Input
# --------------------------------------------------

question = st.text_area(
    "Ask a Question",
    height=120,
    placeholder="e.g. If a policyholder cancels within the first month...",
)

# --------------------------------------------------
# Query Execution
# --------------------------------------------------

if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a question")

    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set")

    else:
        rag_engine = None

        try:
            with st.spinner("Querying Knowledge Graph..."):

                rag_engine = KGRAGEngine(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    database=database,
                    vector_index=vector_index,
                    embedding_model=embedding_model,
                    answer_model=answer_model,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k,
                )

                answer = rag_engine.answer(question)

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.exception(e)

        finally:
            if rag_engine:
                rag_engine.close()
