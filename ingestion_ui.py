import streamlit as st
import os
from dotenv import load_dotenv
import httpx
from pathlib import Path

from kg_ingestion_pipeline import KGIngestionPipeline

# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment")
    st.stop()

# --------------------------------------------------
# Shared HTTP client
# --------------------------------------------------

http_client = httpx.Client(verify=False)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="InsightGraph AI - Ingestion",
    layout="wide",
)

# --------------------------------------------------
# Asset Paths
# --------------------------------------------------

ASSETS = Path(__file__).parent / "assets"
IMPETUS_LOGO = ASSETS / "impetus_logo.png"
NEO4J_LOGO = ASSETS / "neo4j_logo.png"

# --------------------------------------------------
# Header Branding
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if IMPETUS_LOGO.exists():
        st.image(str(IMPETUS_LOGO), width=220)

st.markdown(
    """
    <h2 style='text-align:center; margin-top:0px; margin-bottom:5px;'>
    InsightGraph AI — Data Ingestion
    </h2>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Hover for platform description",
    help="""
Upload PDFs and build a Coarse-Grained Knowledge Graph.

• Creates vector embeddings  
• Builds graph relationships  
• Stores in Neo4j
""",
)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar – Neo4j Configuration
# --------------------------------------------------

st.sidebar.markdown("### Neo4j Configuration")

# Logo ABOVE credentials
if NEO4J_LOGO.exists():
    st.sidebar.image(str(NEO4J_LOGO), width=160)

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

database = st.sidebar.text_input(
    "Database",
    value="insurancedetail",
)

index_name = st.sidebar.text_input(
    "Vector Index",
    value="graph_store",
)

# --------------------------------------------------
# Sidebar – Embedding & Chunking
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Embedding & Chunking")

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["text-embedding-3-small", "text-embedding-3-large"],
)

chunk_size_k = st.sidebar.slider(
    "Chunk Size (k)",
    500,
    2000,
    1000,
    step=100,
)

# --------------------------------------------------
# Sidebar – Relationship Settings
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Relationship Settings")

relationship_similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    0.6,
    0.9,
    0.75,
    step=0.05,
)

relationship_top_k = st.sidebar.slider(
    "Top-K Neighbors per Chunk",
    5,
    50,
    20,
    step=5,
)

# --------------------------------------------------
# Main – PDF Directory
# --------------------------------------------------

st.subheader("PDF Directory")

pdf_dir = st.text_input(
    "Local folder path containing PDFs",
    value=r"D:\DesktopData\KG\Impetus_Coarse_Grained_KG\data",
)

# --------------------------------------------------
# Ingestion Trigger
# --------------------------------------------------

if st.button("Build Knowledge Graph"):

    if not os.path.isdir(pdf_dir):
        st.error("Invalid PDF directory")

    else:
        pipeline = None

        try:
            with st.spinner("Building Knowledge Graph..."):

                pipeline = KGIngestionPipeline(
                    pdf_dir=pdf_dir,
                    neo4j_url=neo4j_uri,
                    neo4j_username=neo4j_user,
                    neo4j_password=neo4j_password,
                    database=database,
                    index_name=index_name,
                    embedding_model=embedding_model,
                    chunk_size_k=chunk_size_k,
                    relationship_similarity_threshold=relationship_similarity_threshold,
                    relationship_top_k=relationship_top_k,
                    http_client=http_client,
                )

                pipeline.run()

            st.success("Knowledge Graph successfully created!")

        except Exception as e:
            st.exception(e)

        finally:
            if pipeline:
                pipeline.close()
