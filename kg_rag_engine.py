"""
KG-RAG Engine (Neo4j + Vector + Graph Traversal)
------------------------------------------------
"""

import os
from typing import List, Dict

import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

from openai_embeddings import OpenAIEmbeddingClient

# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# --------------------------------------------------
# Shared corporate-safe HTTP client
# --------------------------------------------------

http_client = httpx.Client(verify=False)


class KGRAGEngine:

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        database: str,
        vector_index: str,
        embedding_model: str,
        answer_model: str,
        similarity_threshold: float = 0.75,
        top_k: int = 3,
        max_results: int = 10,
    ):
        self.database = database
        self.vector_index = vector_index
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.max_results = max_results

        # Neo4j driver
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )

        # ✅ Embeddings (OpenAI client adapter)
        self.embeddings = OpenAIEmbeddingClient(
            model=embedding_model,
            api_key=OPENAI_API_KEY,
            http_client=http_client,
        )

        # ✅ Chat LLM
        self.answer_llm = ChatOpenAI(
            model=answer_model,
            temperature=0.0,
            api_key=OPENAI_API_KEY,
            http_client=http_client,
            max_retries=5,
        )

    # --------------------------------------------------
    # Cypher Template
    # --------------------------------------------------

    def _cypher_query(self) -> str:
        return """
        CALL db.index.vector.queryNodes(
          $index_name,
          $top_k,
          $query_embedding
        )
        YIELD node AS seed, score
        MATCH (seed)-[r:SIMILAR_TO]-(related:Chunk)
        WHERE r.score >= $similarity_threshold
        RETURN DISTINCT
          seed.text AS seed_text,
          related.text AS related_text,
          r.score AS similarity_score
        ORDER BY r.score DESC
        LIMIT $max_results
        """

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    def retrieve_context(self, question: str) -> List[Dict]:
        query_embedding = self.embeddings.embed_query(question)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                self._cypher_query(),
                index_name=self.vector_index,
                top_k=self.top_k,
                query_embedding=query_embedding,
                similarity_threshold=self.similarity_threshold,
                max_results=self.max_results,
            )

            contexts = []
            for record in result:
                contexts.append(
                    {
                        "seed_text": record["seed_text"],
                        "related_text": record["related_text"],
                        "similarity_score": record["similarity_score"],
                    }
                )

        return contexts

    # --------------------------------------------------
    # Context Assembly
    # --------------------------------------------------

    @staticmethod
    def build_context(contexts: List[Dict]) -> str:
        unique_chunks = set()

        for ctx in contexts:
            unique_chunks.add(ctx["seed_text"])
            unique_chunks.add(ctx["related_text"])

        return "\n\n".join(unique_chunks)

    # --------------------------------------------------
    # Answer Generation
    # --------------------------------------------------

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
You are a domain expert assistant.

Answer the question strictly using the context below.
If the answer is not present in the context, say:
"Answer not found in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.answer_llm.invoke(prompt)
        return response.content.strip()

    # --------------------------------------------------
    # End-to-End RAG
    # --------------------------------------------------

    def answer(self, question: str) -> str:
        contexts = self.retrieve_context(question)

        if not contexts:
            return "No relevant context found."

        context_text = self.build_context(contexts)
        return self.generate_answer(question, context_text)

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def close(self):
        if self.driver:
            self.driver.close()
