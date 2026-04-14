
# 🚀 Hybrid RAG System (FAISS + Elasticsearch + Groq)

## 📌 Overview
This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** system that combines:

- 🔎 **Vector Search (FAISS)** → Semantic understanding  
- 📚 **Keyword Search (Elasticsearch - BM25)** → Exact matching  
- 🤖 **LLM (Groq API)** → Answer generation  

👉 This improves retrieval accuracy compared to traditional RAG.

---

## 🧠 Architecture
User Query
↓
| Hybrid Retrieval |
| ├── FAISS (Vector) |
| └── Elasticsearch |

↓
Context
↓
Groq LLM
↓
Final Answer
-----

## ⚙️ Tech Stack

- Python 3.10
- FAISS (Vector Database)
- Elasticsearch (Keyword Search)
- Groq API (LLM)
- Docker (Containerization)
- LangChain (Data processing)

---

## 🐳 Docker Setup

Start Elasticsearch using Docker:

```bash
docker-compose up -d
Runs Elasticsearch on:

http://localhost:9200

# Create virtual environment
py -3.10 -m venv venv

# Activate
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

hybrid_rag_project/
│
├── app.py
├── ingest.py
├── retriever.py
├── docker-compose.yml
├── requirements.txt
├── data/
│   └── sample.txt
└── .env
