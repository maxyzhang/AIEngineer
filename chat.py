from dotenv import load_dotenv
import os
import json
from memory import (
    load_history, 
    save_history,
    load_memory,
    save_memory
)
from openai_client import get_client
from prompts import SYSTEM_PROMPT
from tools_definitions import TOOLS
from tools import (
    get_current_time, 
    calculate, 
    read_file,
    save_user_memory,
    get_user_memory
    )

TOOL_FUNCTIONS= {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "read_file": read_file,
    "save_user_memory": save_user_memory,
    "get_user_memory": get_user_memory,
}

client = get_client()

history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

history.extend([
    m for m in load_history()
    if m.get("role") != "system"
])

memory = load_memory()
memory_text = f"\n\nLong-term memory:\n{json.dumps(memory, ensure_ascii=False)}"
history[0]["content"] = SYSTEM_PROMPT + memory_text

print("AI Chat starts!")
print("Input exit to quit \n")

while True:
    question = input("You: ")

    if "my name is" in question.lower():
        name = question[11:].strip()
        memory["name"] = name
        save_memory(memory)

    if question.lower() == "exit":
        print("bye! ")
        break
    blocked_requests = [".env", "api key", "apikey", "token", "password"]

    if any(word in question.lower() for word in blocked_requests):
        print("\nAI:")
        print("Access denied.")
        print("-" * 50)
        continue

    history.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=history,
        tools=TOOLS,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        history.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            arguments = json.loads(tool_call.function.arguments or "{}")

            tool_function = TOOL_FUNCTIONS.get(tool_name)

            if tool_function:
                tool_result = tool_function(**arguments)
            else:
                tool_result = "Unknown tool"

            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        second_response = client.chat.completions.create(
            model="gpt-5.5",
            messages=history
        )    

        answer = second_response.choices[0].message.content

    else:
        answer = message.content

    history.append({
        "role": "assistant",
        "content": answer
    })

    save_history(history[1:])

    print("\nAI: ", end="", flush=True)
    print(answer)
    print("-" * 50)
