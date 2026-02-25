# 🧠 Coarse-Grained Knowledge Graph RAG (Neo4j + OpenAI)

This project implements a **Coarse-Grained Knowledge Graph (KG) based RAG system** using:

- Neo4j (Vector Index + Graph Relationships)
- OpenAI Embeddings
- Streamlit UI
- FAISS (for simple vector baseline comparison)
- Python 3.11.5

The goal is to compare:

- ✅ Simple Vector RAG  
- ✅ Neo4j Vector RAG  
- ✅ Graph + Vector (Coarse-Grained KG RAG)  

---

## 📌 What Are We Doing?

We are building a **Graph-Augmented Retrieval System** for PDF documents.

Instead of extracting entities and triples (fine-grained KG), we:

1. Split PDFs into semantic chunks  
2. Store each chunk as a node in Neo4j  
3. Generate embeddings for each chunk  
4. Create relationships between chunks based on similarity threshold  

Example: # 🧠 Coarse-Grained Knowledge Graph RAG (Neo4j + OpenAI)

This project implements a **Coarse-Grained Knowledge Graph (KG) based RAG system** using:

- Neo4j (Vector Index + Graph Relationships)
- OpenAI Embeddings
- Streamlit UI
- FAISS (for simple vector baseline comparison)
- Python 3.11.5

The goal is to compare:

- ✅ Simple Vector RAG  
- ✅ Neo4j Vector RAG  
- ✅ Graph + Vector (Coarse-Grained KG RAG)  

---

## 📌 What Are We Doing?

We are building a **Graph-Augmented Retrieval System** for PDF documents.

Instead of extracting entities and triples (fine-grained KG), we:

1. Split PDFs into semantic chunks  
2. Store each chunk as a node in Neo4j  
3. Generate embeddings for each chunk  
4. Create relationships between chunks based on similarity threshold  

Example:
# 🧠 Coarse-Grained Knowledge Graph RAG (Neo4j + OpenAI)

This project implements a **Coarse-Grained Knowledge Graph (KG) based RAG system** using:

- Neo4j (Vector Index + Graph Relationships)
- OpenAI Embeddings
- Streamlit UI
- FAISS (for simple vector baseline comparison)
- Python 3.11.5

The goal is to compare:

- ✅ Simple Vector RAG  
- ✅ Neo4j Vector RAG  
- ✅ Graph + Vector (Coarse-Grained KG RAG)  

---

## 📌 What Are We Doing?

We are building a **Graph-Augmented Retrieval System** for PDF documents.

Instead of extracting entities and triples (fine-grained KG), we:

1. Split PDFs into semantic chunks  
2. Store each chunk as a node in Neo4j  
3. Generate embeddings for each chunk  
4. Create relationships between chunks based on similarity threshold  

Example: (:Chunk)-[:SIMILAR_TO {score: 0.82}]->(:Chunk)

