"""
Coarse-Grained Knowledge Graph Builder (Neo4j + OpenAI)
------------------------------------------------------
"""

import os
from typing import List

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document

from openai_embeddings import OpenAIEmbeddingClient

# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")


class Neo4jKGBuilder:
    """
    Builds a coarse-grained Knowledge Graph backed by Neo4jVector.
    """

    def __init__(
        self,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        database: str,
        embedding_model: str,
        index_name: str,
    ):
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.database = database
        self.embedding_model = embedding_model
        self.index_name = index_name

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _create_database_if_not_exists(self):
        driver = GraphDatabase.driver(
            self.neo4j_url,
            auth=(self.neo4j_username, self.neo4j_password),
        )

        try:
            with driver.session(database="system") as session:
                session.run(f"CREATE DATABASE {self.database} IF NOT EXISTS")
        finally:
            driver.close()

    def _get_embeddings(self):
        return OpenAIEmbeddingClient(
            model=self.embedding_model,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def build(self, documents: List[Document]) -> Neo4jVector:
        if not documents:
            raise RuntimeError("No documents provided for vector store build")

        self._create_database_if_not_exists()

        embeddings = self._get_embeddings()

        vectorstore = Neo4jVector.from_documents(
            documents=documents,
            embedding=embeddings,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.database,
            index_name=self.index_name,
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
        )

        print(
            f"✅ KG inserted | database={self.database} "
            f"| index={self.index_name} "
            f"| model={self.embedding_model}"
        )

        return vectorstore
