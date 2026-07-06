from openai_client import get_client
from vector_search import search_vector

client = get_client()

print("Interview Coach")
print("Ask an interview question.")
print("Type exit to quit.\n")

while True:
    question = input("Question: ")

    if question.lower() == "exit":
        print("bye!")
        break

    context = search_vector(question)

    prompt = f"""
You are an interview coach for a senior software engineer.

Use the candidate knowledge below to answer the interview question.

Candidate Knowledge:
{context}

Interview Question:
{question}

Generate:
1. A strong interview answer
2. A shorter version
3. STAR structure if applicable
4. Key points to remember

Be specific, technical, confident, and truthful.
Do not invent facts.
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("\nAnswer:\n")
    print(response.choices[0].message.content)
    print("-" * 60)