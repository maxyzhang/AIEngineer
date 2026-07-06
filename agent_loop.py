from openai_client import get_client
from tools.search_tool import run as search_tool 
from tools.calculator_tool import run as calculator_tool

client = get_client()


def call_tool(action, tool_input):
    action = action.lower().strip()

    if action == "search":
        return search_tool(tool_input)

    if action == "calculator":
        return calculator_tool(tool_input)

    return f"Unknown tool: {action}"


def run(question):
    planner_prompt = f"""
You are an AI agent.

Available tools:
1. search - use this for questions about Max, resume, projects, Shell, CDIS, NVIDIA, DICE, interview, career, knowledge base.
2. calculator - use this for math calculations.

User question:
{question}

Decide if a tool is needed.

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

    planner_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": planner_prompt}]
    )

    plan = planner_response.choices[0].message.content.strip()

    print("\n[Plan]")
    print(plan)

    action = "final"
    tool_input = plan

    for line in plan.splitlines():
        if line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()
        elif line.lower().startswith("input:"):
            tool_input = line.split(":", 1)[1].strip()

    if action.lower() == "final":
        return tool_input

    observation = call_tool(action, tool_input)

    print("\n[Observation]")
    print(observation)

    final_prompt = f"""
You are an AI agent.

User question:
{question}

You used this tool:
{action}

Tool observation:
{observation}

Now write the final answer.

Use only the observation when it contains candidate/project facts.
Do not invent facts.
"""

    final_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return final_response.choices[0].message.content
