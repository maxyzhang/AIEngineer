import json
import os
from datetime import datetime
from openai_client import get_client

MEMORY_FILE = "memory.json"
HISTORY_FILE = "history.json"  # agent step history
CONVERSATION_FILE = "conversation..json"  # user chat history

client = get_client()

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


def merge_memory(memory, new_facts):
    if "long_term_memory" not in memory:
        memory["long_term_memory"] = []

    existing = memory["long_term_memory"]

    for fact in new_facts:
        fact = fact.strip()
        if fact and fact not in existing:
            existing.append(fact)

    memory["long_term_memory"] = existing
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