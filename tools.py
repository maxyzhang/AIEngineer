import json
from memory import load_memory, save_memory
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()

BLOCKED_FIELS = {
    ".env",
    ".gitignore",
    "history.json"
}

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression):
    try:
        return str(eval(expression))
    except Exception:
        return "Calculate Error"

def read_file(file_path):
    try:
        path = (PROJECT_ROOT / file_path).resolve()

        if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
            return "Access Denied."
        
        if path.name in BLOCKED_FIELS:
            return "Access Denied."
        
        if path.suffix.lower() not in {
            ".txt",
            ".md",
            ".py",
            ".json"
        }:
            return "Unsuppoted filke type."

        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    except Exception as e:
        return str(e)
    
def save_user_memory(key, value):
    memory = load_memory()
    memory[key] = value
    save_memory(memory)
    return "Memory saved."

def get_user_memory():
    return json.dumps(load_memory(), ensure_ascii=False)