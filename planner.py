from openai_client import get_client

client = get_client()


def create_plan(question, memory_text="", conversation_context=""):
    prompt = f"""
You are an AI agent planner.

Available tools:
- search: use for resume, projects, experience, interview, career, knowledge base questions
- calculator: use for math calculations
- final: use only when no tool is needed

Choose exactly one next action.

User question:
{question}

Long-term memory:
{memory_text}

Recent conversation:
{conversation_context}

Respond ONLY in this format:

Action: search
Input: search query

OR

Action: calculator
Input: math expression

OR

Action: final
Input: final answer
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()