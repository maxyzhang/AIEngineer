from openai_client import get_client
from vector_search import search_vector

client = get_client()


def run(question):
    context = search_vector(question)

    prompt = f"""
You are a career strategy agent for a senior software engineer.

Use the candidate knowledge below to analyze the user's career/job question.

Candidate Knowledge:
{context}

User Request:
{question}

Generate:
1. Role fit analysis
2. Strong matching points
3. Weak or missing areas
4. Projects to emphasize
5. Resume positioning advice
6. Interview preparation points

Be honest, practical, and specific.
Do not invent facts.
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content