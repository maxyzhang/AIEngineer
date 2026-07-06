from agent_loop import run

print("AI Engineer Agent Loop")
print("Type exit to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("bye!")
        break

    answer = run(question)
   
    print("\nAI:\n")
    print(answer)
    print("-" * 60)