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

    print("\nAI: ", end="", flush=True)
    answer = ""

    with client.responses.stream(
        model="gpt-5.5",
        input=history,
        tools=TOOLS
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
                answer += event.delta
    print()

    history.append({
        "role": "assistant",
        "content": answer
    })

    save_history(history)
    print("-" * 50)
