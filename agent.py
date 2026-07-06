from agents.router import route
from agents.interview_agent import run as run_interview 
from agents.resume_agent import run as run_resume 
from vector_search import search_vector 
from openai_client import get_client

client = get_client()

print("AI Engineer Agent")
print("Type exit to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("bye!")
        break

    intent = route(question)
    print(f"[Router] Intent: {intent}")

    if intent == "interview":
        answer = run_interview(question)

    elif intent == "resume":
        answer = run_resume(question)

    else:
        context = search_vector(question)

        prompt = f"""
Use the knowledge below to answer the user question.

Knowledge:
{context}

Question:
{question}

Be accurate and do not invent facts.
"""

        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

    print("\nAI:\n")
    print(answer)
    print("-" * 60)