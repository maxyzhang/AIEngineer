import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
from openai_client import get_client

MEMORY_FILE = "memory.json"
HISTORY_FILE = "history.json"  # agent step history
CONVERSATION_FILE = "conversation..json"  # user chat history

client = get_client()
memory_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    lines = []

    for item in items[-limit:]:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item)
        
        if text:
            lines.append(f"- {text}")

    return "\n".join(lines)

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

Return ONLY JSON in this format:
{{
  "facts": [
    "fact 1",
    "fact 2"
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

    existing_facts = memory["long_term_memory"]

    bad_phrases = [
        "i don't have",
        "not enough information",
        "available observations",
        "appears to",
        "not yet known",
        "cannot confirm",
        "no evidence",
    ]

    for fact in new_facts:
        if not isinstance(fact, str):
            continue

        fact = fact.strip()

        if not fact:
            continue

        fact_lower = fact.lower()

        # 不保存否定、不确定或无价值记忆
        if any(phrase in fact_lower for phrase in bad_phrases):
            continue

        if is_duplicate_memory(
            fact,
            existing_facts,
            similarity_threshold,
        ):
            print(f"[Memory] Skipped duplicate: {fact}")
            continue

        existing_facts.append(fact)
        print(f"[Memory] Added: {fact}")

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