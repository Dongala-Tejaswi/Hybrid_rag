from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from elasticsearch import Elasticsearch

loader = TextLoader("C:\\AIML\\hybrid_rag_project\\data\\sample.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc.page_content for doc in docs]
embeddings = model.encode(texts)

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss.index")

es = Elasticsearch("http://localhost:9200")

for i, text in enumerate(texts):
    es.index(index="rag-index", id=i, document={"content": text})

print("Data indexed successfully")
