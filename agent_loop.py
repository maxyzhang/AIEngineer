from openai_client import get_client
from tools.search_tool import run as search_tool 
from tools.calculator_tool import run as calculator_tool
from memory import load_memory, save_memory

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


def run(question, max_steps=3):
    memory = load_memory()
    memory_text = str(memory)
    history = ""

    for step in range(1, max_steps + 1):
        planner_prompt = f"""
You are a ReAct-style AI agent.

Available tools:
1. search - use for Max, resume, projects, Shell, CDIS, NVIDIA, DICE, interview, career, knowledge base.
2. calculator - use for math calculations.

long-tem memory:
{memory_text}

User question:
{question}

Previous steps and observations:
{history}

Decide the next step.

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

        print(f"\n[Step {step} Plan]")
        print(plan)

        action, tool_input = parse_plan(plan)

        if action.lower() == "final":
            memory["last_question"] = question
            save_memory(memory)
            return tool_input

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

    final_prompt = f"""
You are an AI agent.

User question:
{question}

You have completed these tool steps:
{history}

Now write the final answer.

Use only the observations when they contain candidate/project facts.
Do not invent facts.
"""

    final_response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": final_prompt}]
    )

    answer =  final_response.choices[0].message.content
    memory["last_question"] = question
    save_memory(memory)
    return answer
