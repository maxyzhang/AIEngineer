from tools.search_tool import run as search_tool 
from tools.calculator_tool import run as calculator_tool

print("Tool Agent")
print("Type exit to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("bye!")
        break

    q = question.lower()

    if q.startswith("calculate"):
        expression = question.replace("calculate", "").strip()
        result = calculator_tool(expression)
        print("\nTool Used: Calculator")
        print(result)

    elif "search" in q or "tell me about" in q or "what did" in q:
        result = search_tool(question)
        print("\nTool Used: Search")
        print(result)

    else:
        print("\nNo tool selected yet.")
        print("Try: calculate 12 * 8")
        print("Try: tell me about NVIDIA DICE")

    print("-" * 60)