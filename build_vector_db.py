import os
import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "knowledge"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chromadb_db")

collection = client.get_or_create_collection("knowledge")

#collection.delete(where={})

for filename in os.listdir(KNOWLEDGE_DIR):

    path = os.path.join(KNOWLEDGE_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    enbedding = model.encode(text).tolist()

    collection.add (
        ids=[filename],
        documents=[text],
        embeddings=[enbedding],
        metadatas=[{"file": filename}]
    )

print("Knowledge Base Built Successfully!")