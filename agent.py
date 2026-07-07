from agents.router import route
from agents.interview_agent import run as run_interview
from agents.resume_agent import run as run_resume
from agents.career_agent import run as run_career
from agent_loop import run as run_knowledge
from memory import (
    load_memory, 
    save_memory,
    clear_conversation_memory,
)

print("AI Engineer Agent")
print("Type exit to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "reset memory":
        clear_conversation_memory()
        print("\nAI:\n")
        print("Conversation memory cleared.")
        print("-" * 60)
        continue

    if question.lower() == "exit":
        print("bye!")
        break

    intent = route(question)
    print(f"[Router] Intent: {intent}")

    if intent == "interview":
        answer = run_interview(question)
    elif intent == "resume":
        answer = run_resume(question)
    elif intent == "career":
        answer = run_career(question)
    else:
        answer = run_knowledge(question)

    print("\nAI:\n")
    print(answer)
    print("-" * 60)