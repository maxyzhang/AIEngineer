from openai_client import get_client
from vector_search import search_vector

client = get_client()

print("Resume Generator")
print("Paste job description below.")
print("Type END on a new line when finished.\n")

lines = []

while True:
    line = input()
    if line.strip().upper() == "END":
        break
    lines.append(line)

job_description = "\n".join(lines)

context = search_vector(job_description)

prompt = f"""
You are an expert technical resume writer.

Use the candidate knowledge below to generate a resume tailored to the job description.

Candidate Knowledge:
{context}

Job Description:
{job_description}

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

print("\nGenerated Resume Content:\n")
print(response.choices[0].message.content)