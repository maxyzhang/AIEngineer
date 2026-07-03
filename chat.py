from dotenv import load_dotenv
import os
import json
from memory import load_history, save_history
from openai_client import get_client
from prompts import SYSTEM_PROMPT
from tools_definitions import TOOLS
from tools import get_current_time, calculate

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

            if tool_name == "get_current_time":
                tool_result = get_current_time()
            elif tool_name == "calculate":
                arguments = json.loads(tool_call.function.arguments)
                expression = arguments["expression"]
                tool_result = calculate(expression)
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
