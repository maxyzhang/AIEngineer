from openai_client import get_client

client = get_client()


def create_plan(question, memory_text="", conversation_context=""):

    #print("\n==========================MEMORY============")
    #print(memory_text)
    #print("\n=============================================")
    prompt = f"""
You are an AI agent planner.

If the user's question asks what you remember, or ask about the user's project, profile, preference, or prior context,
prioritize Long-term memory before using search.
If long-term memory directly answers the question, choose final.

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