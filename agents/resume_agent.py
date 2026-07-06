from openai_client import get_client
from vector_search import search_vector

client = get_client()


def run(question):
    context = search_vector(question)

    prompt = f"""
You are an expert technical resume writer.

Use the candidate knowledge below to generate resume content.

Candidate Knowledge:
{context}

User Request:
{question}

Generate:
1. Resume Summary
2. Core Skills
3. Relevant Experience Bullets
4. Suggested Projects to Highlight

Be specific, technical, and achievement-oriented.
Do not invent facts.
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content