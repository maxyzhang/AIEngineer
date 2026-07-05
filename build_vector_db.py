import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "knowledge"
DB_DIR = "chromedb_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks    

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chromadb_db")

collection = client.get_or_create_collection("knowledge")

#collection.delete(where={})

for root, dirs, files in os.walk(KNOWLEDGE_DIR):
    for filename in files:
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(root, filename)
        relative_path = os.path.relpath(path, KNOWLEDGE_DIR)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            enbedding = model.encode(chunk).tolist()

        collection.add (
            ids=[f"{filename}_{i}"],
            documents=[chunk],
            embeddings=[enbedding],
            metadatas=[{
                "file": relative_path,
                "chunk": i
                }]
        )

print("Chunked knowledge Base Built Successfully!")