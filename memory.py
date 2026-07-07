import os
import json

HISTORY_FILE = "history.json"
MEMORY_FILE = "memory.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data or {}   
    except:
        return {}
    
def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)

def add_conversation_turn(user_message, assistant_message):
    memory = load_memory()

    conversations = memory.get("conversation_history", [])

    conversations.append({
        "user": user_message,
        "assistant": assistant_message
    })

    memory["conversation_history"] = conversations[-6:]

    save_memory(memory)


def get_conversation_context():
    memory = load_memory()

    conversations = memory.get("conversation_history", [])

    context = ""

    for turn in conversations:
        context += f"User: {turn['user']}\n"
        context += f"Assistant: {turn['assistant']}\n\n"

    return context