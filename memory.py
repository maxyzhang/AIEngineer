import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
from openai_client import get_client

MEMORY_FILE = "memory.json"
HISTORY_FILE = "history.json"  # agent step history
CONVERSATION_FILE = "conversation.json"  # user chat history

client = get_client()
memory_model = SentenceTransformer("all-MiniLM-L6-v2")

def rank_memories(items):
    normalized_items = []

    for item in items:
        normalized = normalize_memory_item(item)

        if normalized["text"]:
            normalized_items.append(normalized)

    return sorted(
        normalized_items,
        key=lambda item: (
            item["importance"],
            item["access_count"],
            item["created_at"],
        ),
        reverse=True,
    )

def retrieve_memories(question, memory, limit=20):
    if not question or not question.strip():
        return []

    items = memory.get("long_term_memory", [])
    normalized_items = []

    for item in items:
        normalized = normalize_memory_item(item)

        if normalized["text"]:
            normalized_items.append(normalized)

    if not normalized_items:
        return []

    texts = [item["text"] for item in normalized_items]

    embeddings = memory_model.encode(
        [question] + texts,
        normalize_embeddings=True,
    )

    question_embedding = embeddings[0]
    memory_embeddings = embeddings[1:]

    similarities = memory_embeddings @ question_embedding

    ranked_pairs = sorted(
        zip(normalized_items, similarities),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    return [
        {
            **item,
            "similarity": float(similarity),
        }
        for item, similarity in ranked_pairs[:limit]
    ]

def get_relevant_memory_text(
    question,
    memory,
    retrieval_limit=20,
    final_limit=8,
    minimum_similarity=0.30,
):
    retrieved_items = retrieve_memories(
        question,
        memory,
        limit=retrieval_limit,
    )

    scored_items = []

    for item in retrieved_items:
        similarity = float(item.get("similarity", 0.0))
        importance = int(item.get("importance", 5))
        access_count = int(item.get("access_count", 0))

        if similarity < minimum_similarity:
            continue

        final_score = (
            similarity * 0.80
            + (importance / 10.0) * 0.18
            + min(access_count, 10) / 10.0 * 0.02
        )

        scored_items.append(
            {
                **item,
                "final_score": final_score,
            }
        )

    scored_items.sort(
        key=lambda item: item.get("final_score", 0.0),
        reverse=True,
    )

    final_items = scored_items[:final_limit]

    return final_items

def normalize_memory_item(item):
    now = datetime.now().isoformat()

    if isinstance(item, dict):
        return {
            "text": item.get("text", "").strip(),
            "importance": int(item.get("importance", 5)),
            "access_count": int(item.get("access_count", 0)),
            "created_at": item.get("created_at", now),
            "last_accessed": item.get("last_accessed", ""),
        }

    return {
        "text": str(item).strip(),
        "importance": 5,
        "access_count": 0,
        "created_at": now,
        "last_accessed": None,
    }

def clear_conversation_memory():
    save_history([])

def clear_history():
    save_history([])

def load_conversation():
    if not os.path.exists(CONVERSATION_FILE):
        return []
    
    with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_conversation(conversation):
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

def get_memory_text(memory, limit=10):
    items = memory.get("long_term_memory", [])
    ranked_items = rank_memories(items)[:limit]
    
    return "\n".join(
        f"= {item['text']}"
        for item in ranked_items
    )

def extract_memory(question, answer):
    prompt = f"""
You are a memory extraction system.

Extract only durable long-term facts that should be remembered about the user.

Good memories:
- User projects
- User role
- Skills
- Preferences
- Long-term goals
- Important ongoing work

Do not store:
- Temporary questions
- One-time calculations
- Generic assistant answers
- Duplicates

User question:
{question}

Assistant answer:
{answer}

Assign an importance score to every memory.

10 = User identity
9 = Current long-term project
8 = Long-tem goal or profession
7 = Strong preference
6 = Stable skill or knowledge
5 = General background fact

If thr memory describes the user's CURRENT project or product, always assign importance 9.

If a memory describes the user's profession or long-term career, assign importance 8.

Do not assign importance 5 to a current long-term project.

Return ONLY JSON in this format:
{{
  "facts": [
    {{
        "text": "fact 1",
        "importance": 8
    }},
    {{
        "text": "fact 2,
        "importance": 6
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()

    try:
        data = json.loads(text)
        return data.get("facts", [])
    except Exception:
        return []

def cosine_similarity(vector_a, vector_b):
    vetcor_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)

def is_duplicate_memory(
    new_fact,
    existing_facts,
    similarity_threshold=0.88,
):
    if not new_fact or not existing_facts:
        return False

    new_fact_normalized = new_fact.strip().lower()

    # 先做完全相同检查，避免不必要的 embedding
    for existing_fact in existing_facts:
        existing_text = (
            existing_fact.get("text", "")
            if isinstance(existing_fact, dict)
            else str(existing_fact)
        )

        if new_fact_normalized == existing_text.strip().lower():
            return True

    existing_texts = [
        item.get("text", "")
        if isinstance(item, dict)
        else str(item)
        for item in existing_facts
    ]

    existing_texts = [
        text.strip()
        for text in existing_texts
        if text and text.strip()
    ]

    if not existing_texts:
        return False

    embeddings = memory_model.encode(
        [new_fact] + existing_texts,
        normalize_embeddings=True,
    )

    new_embedding = embeddings[0]

    for existing_embedding in embeddings[1:]:
        similarity = float(
            np.dot(new_embedding, existing_embedding)
        )

        if similarity >= similarity_threshold:
            return True

    return False

def merge_memory(
    memory,
    new_facts,
    similarity_threshold=0.8,
):
    if "long_term_memory" not in memory:
        memory["long_term_memory"] = []

    # 先把旧字符串格式统一转换成带 metadata 的 dict
    existing_facts = [
        normalize_memory_item(item)
        for item in memory["long_term_memory"]
    ]

    bad_phrases = [
        "i don't have",
        "not enough information",
        "available observations",
        "appears to",
        "not yet known",
        "cannot confirm",
        "no evidence",
    ]

    for item in new_facts:
        # 兼容旧版字符串和新版 dict
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()

            try:
                importance = int(item.get("importance", 5))
            except (TypeError, ValueError):
                importance = 5
        else:
            text = str(item).strip()
            importance = 5

        if not text:
            continue

        # 限制在 1–10
        importance = max(1, min(10, importance))

        text_lower = text.lower()

        if any(phrase in text_lower for phrase in bad_phrases):
            print(f"[Memory] Skipped low-value memory: {text}")
            continue

        if is_duplicate_memory(
            text,
            existing_facts,
            similarity_threshold,
        ):
            print(f"[Memory] Skipped duplicate: {text}")
            continue

        new_item = {
            "text": text,
            "importance": importance,
            "access_count": 0,
            "created_at": datetime.now().isoformat(),
            "last_accessed": None,
        }

        existing_facts.append(new_item)
        print(
            f"[Memory] Added importance={importance}: {text}"
        )

    memory["long_term_memory"] = existing_facts
    return memory

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "last_question": "",
            "last_answer": "",
            "long_term_memory": [],
            "updated_at": ""
        }

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    memory["updated_at"] = datetime.now().isoformat()

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_memory_item(text):
    memory = load_memory()

    if "long_term_memory" not in memory:
        memory["long_term_memory"] = []

    memory["long_term_memory"].append({
        "text": text,
        "created_at": datetime.now().isoformat()
    })

    save_memory(memory)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_conversation():
    if not os.path.exists(CONVERSATION_FILE):
        return []

    with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_conversation(conversation):
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)


def add_conversation_turn(question, answer):
    conversation = load_conversation()

    conversation.append({
        "question": question,
        "answer": answer,
        "created_at": datetime.now().isoformat()
    })

    save_conversation(conversation)


def get_conversation_context(limit=5):
    conversation = load_conversation()
    recent = conversation[-limit:]

    return "\n".join(
        f"User: {item.get('question', '')}\nAssistant: {item.get('answer', '')}"
        for item in recent
    )