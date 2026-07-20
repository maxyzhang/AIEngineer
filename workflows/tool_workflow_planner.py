import json
from typing import Any, Callable

from workflows.tool_workflow import validate_workflow_steps


WorkflowGenerator = Callable[[str], str]


def build_tool_workflow_prompt(
    question: str,
    *,
    max_steps: int = 6,
) -> str:
    """Build the prompt used to generate a structured tool workflow."""

    return f"""
You are a tool workflow planning agent.

Create a JSON workflow that answers the user's request by calling tools.

Available tools:

1. search
   Input: a natural-language search query
   Use for factual lookup, technical knowledge, projects, resumes,
   interviews, career questions, and knowledge-base retrieval.

2. calculator
   Input: a mathematical expression
   Use only for arithmetic calculations.

Rules:

- Return valid JSON only.
- Do not use Markdown code fences.
- Do not answer the user's question directly.
- Return between 1 and {max_steps} steps.
- Every step must have a unique id.
- Every step must include tool and arguments.input.
- Steps execute sequentially.
- A later step may reference a prior result using:
  {{{{step_id.result}}}}
- References may only point to earlier steps.
- Use lowercase tool names.
- Do not invent unsupported tools.
- Keep search queries specific and concise.
- Break calculations into multiple steps when one result is needed later.

Required JSON format:

{{
  "steps": [
    {{
      "id": "step_1",
      "tool": "search",
      "arguments": {{
        "input": "search query"
      }}
    }}
  ]
}}

User request:

{question}
""".strip()


def parse_tool_workflow_response(
    response_text: str,
    *,
    max_steps: int = 6,
) -> dict[str, Any]:
    """Parse and validate an LLM-generated workflow response."""

    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        return {
            "status": "invalid",
            "steps": [],
            "errors": [
                f"Workflow response is not valid JSON: {error.msg}"
            ],
            "raw_response": response_text,
        }

    if not isinstance(payload, dict):
        return {
            "status": "invalid",
            "steps": [],
            "errors": [
                "Workflow response must be a JSON object"
            ],
            "raw_response": response_text,
        }

    steps = payload.get("steps")

    validation_errors = validate_workflow_steps(
        steps,
        max_steps=max_steps,
    )

    if isinstance(steps, list):
        validation_errors.extend(
            validate_supported_tools_and_references(steps)
        )

    if validation_errors:
        return {
            "status": "invalid",
            "steps": steps if isinstance(steps, list) else [],
            "errors": validation_errors,
            "raw_response": response_text,
        }

    return {
        "status": "valid",
        "steps": steps,
        "errors": [],
        "raw_response": response_text,
    }


def validate_supported_tools_and_references(
    steps: list[dict[str, Any]],
) -> list[str]:
    """Validate supported tools and backward-only step references."""

    errors: list[str] = []
    supported_tools = {"search", "calculator"}
    available_step_ids: set[str] = set()

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        step_id = step.get("id")
        tool = step.get("tool")
        arguments = step.get("arguments")

        if isinstance(tool, str):
            normalized_tool = tool.strip().lower()

            if normalized_tool not in supported_tools:
                errors.append(
                    f"Unsupported tool in steps[{index}]: {tool}"
                )

        if isinstance(arguments, dict):
            tool_input = arguments.get("input")

            if isinstance(tool_input, str):
                referenced_ids = extract_step_references(tool_input)

                for referenced_id in referenced_ids:
                    if referenced_id not in available_step_ids:
                        errors.append(
                            "Step reference must point to an earlier "
                            f"step: {referenced_id}"
                        )

        if isinstance(step_id, str) and step_id.strip():
            available_step_ids.add(step_id)

    return errors


def extract_step_references(value: str) -> list[str]:
    """Extract step IDs from {{step_id.result}} references."""

    from workflows.tool_workflow import REFERENCE_PATTERN

    return REFERENCE_PATTERN.findall(value)


def default_workflow_generator(prompt: str) -> str:
    """Generate a workflow using the configured OpenAI client."""

    from openai_client import get_client

    client = get_client()

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={
            "type": "json_object",
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The workflow planner returned an empty response"
        )

    return content


def create_tool_workflow(
    question: str,
    *,
    max_steps: int = 6,
    workflow_generator: WorkflowGenerator = default_workflow_generator,
) -> dict[str, Any]:
    """Generate and validate a structured workflow for a question."""

    if not isinstance(question, str) or not question.strip():
        return {
            "status": "invalid",
            "steps": [],
            "errors": [
                "Question must be a non-empty string"
            ],
            "raw_response": "",
        }

    prompt = build_tool_workflow_prompt(
        question,
        max_steps=max_steps,
    )

    try:
        response_text = workflow_generator(prompt)
    except Exception as error:
        return {
            "status": "failed",
            "steps": [],
            "errors": [str(error)],
            "raw_response": "",
        }

    return parse_tool_workflow_response(
        response_text,
        max_steps=max_steps,
    )

