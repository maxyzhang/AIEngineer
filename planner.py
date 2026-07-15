from openai_client import get_client

client = get_client()

def format_memory_context(memories):
    if not memories:
        return "No relevant long-term memory found."

    # 兼容以前直接传字符串的情况
    if isinstance(memories, str):
        return memories

    lines = []

    for item in memories:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            importance = item.get("importance", 5)
            similarity = float(item.get("similarity", 0.0))

            if text:
                lines.append(
                    f"- {text} "
                    f"(importance={importance}, relevance={similarity:.2f})"
                )
        else:
            text = str(item).strip()

            if text:
                lines.append(f"- {text}")

    return "\n".join(lines) or "No relevant long-term memory found."

def create_plan(
    question, 
    memories=None, 
    conversation_context="",
):
    memory_context = format_memory_context(memories)

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
{memory_context}

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