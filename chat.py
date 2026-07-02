from dotenv import load_dotenv
import os
from memory import load_history, save_history
from openai_client import get_client
from prompts import SYSTEM_PROMPT

client = get_client()

history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]
history.extend(load_history())

print("AI Chat starts!")
print("Input exit to quit \n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("bye! ")
        break
    history.append({
        "role": "user",
        "content": question
    })

    response = client.responses.create(
        model="gpt-5.5",
        input=history
    )

    answer = response.output_text

    history.append({
        "role": "assistant",
        "content": answer
    })

    save_history(history)
    print("\nAI: ")
    print(answer)
    print("-" * 50)
