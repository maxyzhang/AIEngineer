import os
import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "knowledge"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chromadb_db")
collection = client.get_or_create_collection("knowledge")

def detect_folder(question):
    q = question.lower()

    if "node export" in q or "cdis" in q:
        return os.path.join("knowledge", "projects", "cdis")
    
    if "nvidia" in q or "cuda" in q or "dice" in q or "index" in q:
        return os.path.join("knowledge", "projects", "nvidia")
    
    if "linux" in q or "rhel" in q or "migration" in q:
        return os.path.join("knowledge", "projects", "linux_migration")
    
    if "shell" in q or "company" in q:
        return os.path.join("knowledge", "companies")
    
    if "skill" in q or "python" in q or "mongodb" in q or "oracle" in q:
        return os.path.join("knowledge", "skills")
    
    if "education" in q or "degree" in q or "university" in q:
        return os.path.join("knowledge", "education")
    
    if any(word in q for word in ["interview", "tell me about yourself", "leadership", "collaboration", "challenge"]):
        return os.path.join("knowledge", "interview")
    
    return None


def keyword_search(question):
    hits = []
    keywords = question.lower().split()
    target_folder = detect_folder(question)

    for root, dirs, files in os.walk(KNOWLEDGE_DIR):
        if target_folder and not root.startswith(target_folder):
            continue

        for filename in files:
            if not filename.endswith(".txt"):
                continue

            path = os.path.join(root, filename)
            relative_path = os.path.relpath(path, KNOWLEDGE_DIR)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            score = 0
            text_lower = text.lower()

            for word in keywords:
                if word in text_lower:
                    score += 1

            if score > 0:
                hits.append((score, relative_path, text))

        hits.sort(reverse=True)
        return hits[:3]    


def search_vector(question, top_k=15):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    #print("="*60)
    #print(results)
    #print("="*60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # print("DEBUG files:", [m.get("file") for m in metadatas])
    output = ""
    sources = []

    for doc, meta, distance in zip(documents, metadatas, distances):
        file = meta.get("file", "unkown")
        chunk = meta.get("chunk", "old")

        if distance > 1.3:
            continue

        if distance < 0.9:
            confidence = "high"
        elif distance < 1.2:
            confidence = "medium"
        else:
            confidence = "low"

        sources.append(file)

        output += f"\n---{meta['file']} | chunk: {chunk} | distance: {distance: .3f} | confidence: {confidence} ---\n"
        output += doc + "\n"

    output += "\nKeyword Search Results:\n"
    keyword_hits = keyword_search(question) or []

    for score, file, text in keyword_hits:
        sources.append(file)

        output += f"\n--- Source: {file} | keyword score: {score}---\n"
        output += text + "\n"

    unique_sources = list(dict.fromkeys(sources))

    output += "\n\nSources:\n"
    for source in unique_sources:
        output += f"- {source}\n"

    low_count = output.count("confidence: low")
    medium_count = output.count("confidence: medium")
    high_count = output.count("confidence: high")

    output += "\nSearch Confidence Summary:\n"
    output += f"- High confidence results: {high_count}\n"
    output += f"- Medium confidence results: {medium_count}\n"
    output += f"- Low confidence results: {low_count}\n"

    if low_count > high_count + medium_count:
        output += "_ Overall search quality: LOW\n"
    else:
        output += "_ Overall search quality: ACCEPTABLE\n"
    
    question_lower = question.lower()
    output_lower = output.lower()

    coverage_topics = []

    if "cuda" in question_lower:
        coverage_topics.append("CUDA")

    if "rocm" in question_lower:
        coverage_topics.append("ROCm")

    output += "\nCoverage Summary:\n"

    missing_topics = []

    for topic in coverage_topics:
        if topic.lower() in output_lower:
            output += f"- {topic}: FOUND\n"
        else:
            output += f"- {topic}: MISSING\n"
            missing_topics.append(topic)

    if len(coverage_topics) >= 2:
        if missing_topics:
            output += "- Comparison: INCOMPLETE\n"
        else:
            output += "- Comparison: COMPLETE\n"

    return output