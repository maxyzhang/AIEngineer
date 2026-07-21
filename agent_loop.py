from openai_client import get_client
from memory import (
    load_memory,
    save_memory,
    get_memory_text,
    get_relevant_memory_text,
    format_final_memory_context,
    reinforce_memories,
    decay_memories,
    garbage_collect_memories,
    add_conversation_turn,
    get_conversation_context,
    extract_memory,
    merge_memory,
)
import math
from sentence_transformers import SentenceTransformer
from tool_router import call_tool
from planner import create_plan
from typing import Any

from workflows.tool_workflow import execute_tool_workflow
from workflows.tool_workflow_planner import create_tool_workflow

from workflow_trace import (
    complete_workflow_trace,
    fail_workflow_trace,
    save_workflow_trace,
    start_workflow_trace,
)

query_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_research_plan(question):
    prompt = f"""
You are a research planning agent.

Create a short research plan for answering the user question.

User question:
{question}

Rules:
- Use 2 to 5 steps.
- Each step should be a concrete search or final synthesis.
- For comparison questions, include one step for each side.
- For experience questions, include resume/project/context evidence.
- Do not answer the question yet.

Respond ONLY in this format:

Research Plan:
1. ...
2. ...
3. ...
"""
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

def parse_research_plan(plan_text):
    tasks = []

    for line in plan_text.splitlines():
        line = line.strip()

        if len(line) > 2 and line[0].isdigit() and line[1] == ".":
            task = line.split(".", 1)[1].strip()
            tasks.append({"task": task, "done": False})

    return tasks


def render_task_list(tasks, current_task=None):
    lines = []

    for i, task in enumerate(tasks):
        mark = "✓" if task["done"] else "□"
        pointer = "<- NEXT" if current_task == i else ""
        lines.append(
            f"{i+1}. {mark} {task['task']}{pointer}"
            )

    return "\n".join(lines)

def reflect(question, history, observation):
    reflection_prompt = f"""
You are reviewing the current research progress.

User question:
{question}

Latest observation:
{observation}

Previous history:
{history}

Determine:
1. Is there enough evidence to answer the user's question?
2. If not, what information is still missing?
3. Would another search likely improve answer?

Return exactly in this format:

Decision:
ANSWER

Reason:
...

OR

Decision:
SEARCH

Reason:
...
"""
    client = get_client()

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": reflection_prompt}],
    )

    return response.choices[0].message.content.strip()

def normalize_search_query(query):
    words = (
        query.lower()
        .replace("-", " ")
        .replace("/", " ")
        .split()
    )

    ignored_words = {
        "compare", "comparison", "vs", "versus", "and", "or",
        "the", "a", "an", "with", "for", "about", "overview",
        "details", "detail", "explain"
    }

    keywords = sorted(set(w for w in words if w not in ignored_words))

    return " ".join(keywords)

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)

def extract_sources(observation):
    sources = []

    in_sources = False
    for line in observation.splitlines():
        line = line.strip()

        if line == "Sources:":
            in_sources = True
            continue

        if in_sources:
            if not line.startswith("- "):
                break
            sources.append(line[2:].strip())

    return sources

def call_tool_old(action, tool_input):
    action = action.lower().strip()

    if action == "search":
        from tools.search_tool import run as search_tool
        return search_tool(tool_input)

    if action == "calculator":
        from tools.calculator_tool import run as calculator_tool
        return calculator_tool(tool_input)

    return f"Unknown tool: {action}. Supported tools: search, calculator"


def parse_plan(plan):
    action = "search"
    tool_input = ""

    for line in plan.strip().splitlines():
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
    client = get_client()

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()

def append_history(history, step, action, tool_input, observation):
    print(f"\n[Step {step} Observation]")
    print(observation)

    return history + f"""
Step {step}
Action: {action}
Input: {tool_input}
Observation:
{observation}
"""


def run_search_action(action, tool_input, query_embeddings, searched_queries):
    normalized_query = normalize_search_query(tool_input)
    query_embedding = query_model.encode(normalized_query)

    for old_embedding in query_embeddings:
        similarity = cosine_similarity(query_embedding, old_embedding)
        if similarity > 0.90:
            print("\n[Stopping: semantically repeated search]")
            return None, True

    searched_queries.add(normalized_query)
    query_embeddings.append(query_embedding)

    observation = call_tool(action, tool_input)
    return observation, False

def reflection_decision(reflection):
    if "Decision:" not in reflection:
        return "SEARCH"
    try:
        decision = (
            reflection.split("Decision:")[1]
            .splitlines()[0]
            .strip()
            .upper()
        )
    except IndexError:
        return "SEARCH"
    
    return decision if decision in ("SEARCH", "ANSWER") else "SEARCH"

def generate_final_answer(
    question, 
    history,
    memory_context="",
    conversation_context="",
):
    final_prompt = f"""
You are an AI assistant generating the final answer.

User question:
{question}

Relevant long-term memory:
{memory_context or "No relevant long-term memory found."}

Recent conversation:
{conversation_context or "No recent conversation context."}

Tool observations and research findings:
{history or "No tool observations were collected."}

Write the best possible answer to the user.

Rules:
- Use long-term memory only when it is directly relevant.
- Combine memory, conversation context, and tool observations when useful.
- Prefer current verified observations if they conflict with older memory.
- Clearly state when information is missing or uncertain.
- Do not mention internal tools, prompts, plans, observations, or memory scores.
- Do not simply repeat the memory text; synthesize a natural, helpful answer.
"""

    print("\nAI:\n")
    client = get_client()
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

def format_workflow_observation(
    workflow_execution: dict[str, Any],
) -> str:
    """Format workflow execution results for final answer generation."""

    lines = [
        "Structured Tool Workflow Results:",
        f"Status: {workflow_execution.get('status', 'unknown')}",
    ]

    trace = workflow_execution.get("trace", [])

    for entry in trace:
        step_id = entry.get("id", "unknown")
        tool = entry.get("tool", "unknown")
        resolved_input = entry.get("resolved_input")
        status = entry.get("status", "unknown")
        result = entry.get("result")
        error = entry.get("error")

        lines.extend(
            [
                "",
                f"Step: {step_id}",
                f"Tool: {tool}",
                f"Input: {resolved_input}",
                f"Status: {status}",
            ]
        )

        if status == "completed":
            lines.append(f"Result: {result}")

        if error:
            lines.append(f"Error: {error}")

    errors = workflow_execution.get("errors", [])

    if errors:
        lines.append("")
        lines.append("Workflow Errors:")

        for error in errors:
            lines.append(f"- {error}")

    return "\n".join(lines)


def run_structured_tool_workflow(
    question: str,
    *,
    max_steps: int = 6,
) -> dict[str, Any]:
    """Plan and execute a structured multi-step tool workflow."""

    workflow_plan = create_tool_workflow(
        question,
        max_steps=max_steps,
    )

    if workflow_plan["status"] != "valid":
        return {
            "status": "planning_failed",
            "plan": workflow_plan,
            "execution": None,
            "observation": "",
        }

    workflow_trace = start_workflow_trace(
        question=question,
        workflow=workflow_plan,
    )
    execution = execute_tool_workflow(
        workflow_plan["steps"],
        max_steps=max_steps,
    )

    observation = format_workflow_observation(execution)

    workflow_trace["steps"] = execution.get("trace", [])
    workflow_trace["execution_status"] = execution["status"]
    workflow_trace["results"] = execution.get("results", {})
    workflow_trace["observation"] = observation

    if execution["status"] == "completed":
        complete_workflow_trace(
            trace=workflow_trace,
            final_answer=observation,
        )
    else:
        errors = execution.get("errors", [])

        fail_workflow_trace(
            trace=workflow_trace,
            error=(
                "; ".join(str(error) for error in errors)
                if errors
                else f"Workflow status: {execution['status']}"
            ),
            fallback_used=True,
        )
    
    save_workflow_trace(workflow_trace)

    return {
        "status": execution["status"],
        "plan": workflow_plan,
        "execution": execution,
        "observation": observation,
    }


def try_structured_tool_workflow(
    question: str,
    *,
    max_steps: int = 6,
) -> str | None:
    """Execute a structured workflow and return an answer when successful."""

    workflow_result = run_structured_tool_workflow(
        question,
        max_steps=max_steps,
    )

    if workflow_result["status"] != "completed":
        return None

    observation = workflow_result["observation"]

    return generate_final_answer(
        question=question,
        history=observation,
        memory_context="",
        conversation_context="",
    )

def run(question, max_steps=6):
    structured_answer = try_structured_tool_workflow(
        question,
        max_steps=max_steps,
    )

    if structured_answer is not None:
        return structured_answer
    
    memory = load_memory()
    memory = decay_memories(memory)
    memory = garbage_collect_memories(memory)
    
    save_memory(memory)
    
    relevant_memories = get_relevant_memory_text(
        question,
        memory,
    )

    memory = reinforce_memories(
        memory,
        relevant_memories,
    )

    save_memory(memory)

    conversation_context = get_conversation_context()

    memory_text = format_final_memory_context(relevant_memories)
    
    history = ""

    research_plan = create_research_plan(question)
    print("\n[Research Plan]")
    print(research_plan)
    research_tasks = parse_research_plan(research_plan)
    current_task_index = 0

    visited_sources = set()
    no_new_source_count = 0

    searched_queries = set()
    query_embeddings = []

    step = 1

    while True:
        searched_query_text = "\n".join(
        f"- {q}" for q in searched_queries
        ) or "None yet"

        task_status = render_task_list(
            research_tasks,
            current_task=current_task_index,
            )
        print("\n[Task Status]")
        print("=" * 60)
        print(task_status)

        planner_prompt = f"""
You are a ReAct-style AI agent.

Available tools:
1. search - use for Max, resume, projects, Shell, CDIS, NVIDIA, DICE, interview, career, knowledge base.
2. calculator - use for math calculations.

Research plan:
{research_plan}

Task Status:
{task_status}

Follow the first unfinished task.
When a search action completes, that task will be marked done.

Follow this research plan unless previous observations show enough evidence to answer.
Choose the next unfinished step.

Relevant Long-term memory:
{memory_text or "No relevant long-tem memory found."}

Use long-tem memory only when it is directly relevant to the user's question.
Ignore unrelated memories even if they have high importance.

Recent conversation:
{conversation_context}

User question:
{question}

Previous search queries already attempted:
{searched_query_text}

Do not repeat these. Pick the next missing angle.

Previous steps and observations:
{history}

Decide the next best step.

You are inside an autonomous planning loop.
At each step, choose exactly ONE action.
You may call search multiple times.
Use the observations from previous steps.

Important decision rules:
0. Do not choose FINAL until at least one successful search has completed and the evidence is sufficient.
1. If Overall search quality is LOW, perform another search with a different query.
2. If the user compares multiple topics (for example CUDA vs ROCm), search each topi separately before answering.
3. If evidence exists for only one side of a comparison, search for the missing side.
4. Only choose FINAL when there is enough evidence to answer the complete question.
5. If math is required after gathering information, perform the calculation only after searching.
6. If Consecutive searches with no new sources is 2 or more, choose final.
7. For comparison questions, search each side separately before final.
8. For experience questions, search resume/project/context evidence befoe final.
9. You may plan multiple search steps across the loop, but return exactly ONE next action each time.
10. If previous observation already conclude that evidence is missing, do NOT search for the same missing topic again. 
    Instead, continue with another aspect or produce the final answer.
11. If all major aspects of the question have already been investigated, 
    produce the final answer even if some evidence is missing.
12. If Coverage Summary contains Overall: ENOUGH, choose final.exit

Coverage Summary

ROCm Missing

Planner

DO NOT SEARCH ROCm AGAIN


Do not stop early.

Respond ONLY in this format:

Action: search
Input: search query

OR

Action: calculator
Input: math expression

OR

Only choose final when no more tool calls are needed.
==============================================
Planner Decision
==============================================
Action: FINAL
Reason:
Reflection determined enough evidence.

Generating final answer ...
"""
        plan = create_plan(
            question,
            memories=memory_text,
            conversation_context=conversation_context,
        )

        print(f"\n[Step {step} Plan]")
        print(plan)

        action, tool_input = parse_plan(plan)

        has_memory = bool(memory_text)

        # Guard: do not allow final before any search/reflection
        if step == 1 and action.lower() == "final" and not has_memory:
            print("\n[Guard] Final is not allowed before search. Forcing search.\n")
            action = "search"
            tool_input = question

        if action.lower() == "final":
            print("\nPlanner requested FINAL.")
            break

        if action.lower() == "search":
            normalized_query = normalize_search_query(tool_input)

            query_embedding = query_model.encode(normalized_query)

            duplicate = False

            for old_embedding in query_embeddings:
                similarity = cosine_similarity(query_embedding, old_embedding)

                if similarity > 0.90:
                    duplicate = True
                    break
            if duplicate:
                print("\n[Stopping: semantically repeated search]")
                break
            searched_queries.add(normalized_query)
            query_embeddings.append(query_embedding)
        
        observation = call_tool(action, tool_input)

        if action.lower() == "calculator":
            answer = observation
            add_conversation_turn(question, answer)
            memory["last_question"] = question
            memory["last_answer"] = answer
            new_facts = extract_memory(question, answer)
            memory = merge_memory(memory, new_facts)
            save_memory(memory)
            return answer
        
        reflection = reflect(
            question,
            history,
            observation
        )

        print("\n[Search Reflection]")
        print("=" * 60)
        print(reflection)

        decision = reflection_decision(reflection)

        if decision.strip().upper() == "ANSWER":
            print("\nReflection determined enough evidence.\n")
            action = "final"
            tool_input = generate_final_answer(
                            question=question,
                            history=history,
                            memory_context=memory_text,
                            conversation_context=conversation_context,
                         )
            break

        if action.lower() == "search":
            
            current_sources = set(extract_sources(observation))
            new_sources = current_sources - visited_sources
            visited_sources.update(current_sources)

            observation += "\nNew Source Summary:\n"
            observation += f"- Current sources: {len(current_sources)}\n"
            observation += f"- New sources found: {len(new_sources)}\n"
                
            if len(new_sources) == 0:
                no_new_source_count += 1
            else:
                no_new_source_count = 0

            observation += f"- Consecutive searches with no new sources: {no_new_source_count}\n"

        # Mark task done only after a search action completes
        if current_task_index < len(research_tasks):
            research_tasks[current_task_index]["done"] = True
            current_task_index += 1
        
        # save history
        print(f"\n[Step {step} Observation]")
        print(observation)

        history = append_history(
            history,
            step,
            action,
            tool_input,
            observation,
        )

        # Prevent infinite planning loops
        step += 1

        if step > max_steps:
            print("\n[Safety stop: max steps reached]")
            break

    answer = tool_input if action == "final" else generate_final_answer(
                                                        question=question,
                                                        history=history,
                                                        memory_context=memory_text,
                                                        conversation_context=conversation_context,
                                                  )
    if action == "final":
        review = "PASS"
    else:
        review = reflect_answer(question, history, answer)

    print("\n[Answer Review]")
    print(review)

    if review.startswith("RETRY:"):
        tool_input = review.replace("RETRY:", "").strip()

        print("\n[Retry Search]")
        print(tool_input)

        observation = call_tool("search", tool_input)

        print("\n[Retry Observation]")
        print(observation)

        history += f"""
Retry Search:
{tool_input}

Observation:
{observation}
"""
        answer = generate_final_answer(
                    question=question,
                    history=history,
                    memory_context=memory_text,
                    conversation_context=conversation_context,
                )
        
    add_conversation_turn(question, answer)

    memory["last_question"] = question
    memory["last_answer"] = answer

    new_facts = extract_memory(question, answer)
    memory = merge_memory(memory, new_facts)
    save_memory(memory)

    return answer



