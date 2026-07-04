import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chromadb_db")
collection = client.get_or_create_collection("knowledge")

def search_vector(question, top_k=3):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = ""

    for doc, meta, distance in zip(documents, metadatas, distances):
        output += f"\n---{meta['file']} | distance: {distance} ---\n"
        output += doc + "\n"

    return output