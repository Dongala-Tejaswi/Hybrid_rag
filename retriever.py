import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

index = faiss.read_index("faiss.index")
model = SentenceTransformer("all-MiniLM-L6-v2")
es = Elasticsearch("http://localhost:9200")

def hybrid_search(query):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=5)

    res = es.search(index="rag-index", query={
        "match": {"content": query}
    })

    keyword_results = [hit["_source"]["content"] for hit in res["hits"]["hits"]]

    return I[0], keyword_results
