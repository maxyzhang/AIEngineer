from openai_client import get_client
from tools.search_tool import run as search_tool 
from tools.calculator_tool import run as calculator_tool 
from memory import (
    load_memory,
    save_memory,
    add_conversation_turn,
    get_conversation_context,
)

client = get_client()


def call_tool(action, tool_input):
    action = action.lower().strip()

    if action == "search":
        return search_tool(tool_input)

    if action == "calculator":
        return calculator_tool(tool_input)

    return f"Unknown tool: {action}"


def parse_plan(plan):
    action = "final"
    tool_input = plan

    for line in plan.splitlines():
        if line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()
        elif line.lower().startswith("input:"):
            tool_input = line.split(":", 1)[1].strip()

    return action, tool_input


def reflect_answer(question, history, answer):
    prompt = f"""
You are a strict AI answer reviewer.

User question:
{question}

Tool observations:
{history}

Draft answer:
{answer}

Check if the answer is grounded in the observations.

Respond ONLY with:

PASS

or

RETRY: search query
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def generate_final_answer(question, history):
    final_prompt = f"""
You are an AI assistant.

User question:
{question}

Updated observations:
{history}

Write the final answer using only the observations.
Do not invent facts.
"""

    print("\nAI:\n")

    stream = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": final_prompt}],
        stream=True,
    )

    answer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            answer += delta

    print()
    return answer


def run(question, max_steps=3):
    memory = load_memory()
    memory_text = str(memory)
    conversation_context = get_conversation_context()

    history = ""

    for step in range(1, max_steps + 1):
        planner_prompt = f"""
You are a ReAct-style AI agent.

Available tools:
1. search - use for Max, resume, projects, Shell, CDIS, NVIDIA, DICE, interview, career, knowledge base.
2. calculator - use for math calculations.

Long-term memory:
{memory_text}

Recent conversation:
{conversation_context}

User question:
{question}

Previous steps and observations:
{history}

Decide the next best step.

You may call multiple tools across multiple steps.
Do NOT choose final until you have enough observations to answer.
If the question compares multiple things, search each thing separately.
If the question requires math after search, search first, then calculate.

Respond ONLY in this format:

Action: search
Input: search query

OR

Action: calculator
Input: math expression

OR

Only choose final when no more tool calls are needed.
Action: final
Input: done
"""

        planner_response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": planner_prompt}],
        )

        plan = planner_response.choices[0].message.content.strip()

        print(f"\n[Step {step} Plan]")
        print(plan)

        action, tool_input = parse_plan(plan)

        if action.lower() == "final":
            break

        observation = call_tool(action, tool_input)

        print(f"\n[Step {step} Observation]")
        print(observation)

        history += f"""
Step {step}
Action: {action}
Input: {tool_input}
Observation:
{observation}
"""

    answer = generate_final_answer(question, history)

    review = reflect_answer(question, history, answer)

    print("\n[Reflection]")
    print(review)

    if review.startswith("RETRY:"):
        tool_input = review.replace("RETRY:", "").strip()

        print("\n[Retry Search]")
        print(tool_input)

        observation = call_tool("search", tool_input)

        print(observation)

        history += f"""
Retry Search:
{tool_input}

Observation:
{observation}
"""

        answer = generate_final_answer(question, history)

    add_conversation_turn(question, answer)

    memory["last_question"] = question
    save_memory(memory)

    return answer
