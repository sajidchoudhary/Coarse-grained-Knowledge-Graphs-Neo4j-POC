"""
Coarse-Grained Knowledge Graph Ingestion Pipeline
-------------------------------------------------
- Load PDFs from directory
- Chunk content into LangChain Documents
- Add metadata
- Store embeddings in Neo4j vector store
- Create SIMILAR_TO relationships between chunks
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from neo4j import GraphDatabase
from openai import OpenAI

from pdf_chunker import PDFChunker
from neo4j_vectorstore_builder import Neo4jKGBuilder

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")


class KGIngestionPipeline:
    """
    Orchestrates PDF → Chunk → Neo4j ingestion + graph relationship creation.
    """

    def __init__(
        self,
        pdf_dir: str,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        database: str,
        index_name: str,
        embedding_model: str,
        chunk_size_k: int,
        relationship_similarity_threshold: float,
        relationship_top_k: int = 20,
        http_client=None,  # ← NEW
    ):
        self.pdf_dir = pdf_dir
        self.chunk_size_k = chunk_size_k
        self.database = database
        self.index_name = index_name
        self.relationship_similarity_threshold = relationship_similarity_threshold
        self.relationship_top_k = relationship_top_k

        self.chunker = PDFChunker()

        # ✅ OpenAI client (corporate safe)
        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client,
        )

        # Neo4j KG builder
        self.kg_builder = Neo4jKGBuilder(
            neo4j_url=neo4j_url,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            database=database,
            index_name=index_name,
            embedding_model=embedding_model,
        )

        # Neo4j driver
        self.driver = GraphDatabase.driver(
            neo4j_url,
            auth=(neo4j_username, neo4j_password),
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _load_pdfs_as_documents(self) -> List[Document]:
        documents: List[Document] = []

        if not os.path.isdir(self.pdf_dir):
            raise RuntimeError(f"Invalid PDF directory: {self.pdf_dir}")

        for file in os.listdir(self.pdf_dir):
            if not file.lower().endswith((".pdf", ".odt")):
                continue

            file_path = os.path.join(self.pdf_dir, file)
            print(f"📄 Processing: {file}")

            chunked_docs = self.chunker.extract(
                pdf_path=file_path,
                k=self.chunk_size_k,
            )

            for doc in chunked_docs:
                doc.metadata["source"] = file
                documents.append(doc)

        return documents

    def _create_chunk_relationships(self):
        print(
            f"🔗 Creating relationships "
            f"(threshold={self.relationship_similarity_threshold}, "
            f"top_k={self.relationship_top_k})"
        )

        cypher = """
        MATCH (c:Chunk)
        CALL db.index.vector.queryNodes(
          $index_name,
          $top_k,
          c.embedding
        )
        YIELD node AS other, score
        WHERE
          score >= $threshold
          AND elementId(c) < elementId(other)
        MERGE (c)-[:SIMILAR_TO {score: score}]->(other)
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                cypher,
                index_name=self.index_name,
                threshold=self.relationship_similarity_threshold,
                top_k=self.relationship_top_k,
            )

        print("✅ Chunk relationships created")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run(self):
        print("🚀 Starting KG ingestion pipeline")

        documents = self._load_pdfs_as_documents()
        if not documents:
            raise RuntimeError("No documents found for ingestion")

        print(f"✅ Loaded {len(documents)} chunks")

        # 1. Insert embeddings into Neo4j
        self.kg_builder.build(documents)

        # 2. Create similarity relationships
        self._create_chunk_relationships()

        print("🎉 Coarse-Grained Knowledge Graph successfully created")

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def close(self):
        if self.driver:
            self.driver.close()
