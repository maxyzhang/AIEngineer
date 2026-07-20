import re
from typing import Any, Callable

from tool_router import call_tool


ToolCaller = Callable[[str, str], Any]

REFERENCE_PATTERN = re.compile(
    r"\{\{([A-Za-z_][A-Za-z0-9_-]*)\.result\}\}"
)


def validate_workflow_steps(
    steps: Any,
    max_steps: int = 10,
) -> list[str]:
    """Return validation errors for a multi-step tool workflow."""

    errors: list[str] = []

    if not isinstance(steps, list):
        return ["Workflow steps must be a list"]

    if not steps:
        return ["Workflow must contain at least one step"]

    if len(steps) > max_steps:
        errors.append(
            f"Workflow contains {len(steps)} steps, "
            f"but the maximum is {max_steps}"
        )

    seen_ids: set[str] = set()

    for index, step in enumerate(steps):
        step_path = f"steps[{index}]"

        if not isinstance(step, dict):
            errors.append(f"{step_path} must be a dictionary")
            continue

        step_id = step.get("id")
        tool = step.get("tool")
        arguments = step.get("arguments")

        if not isinstance(step_id, str) or not step_id.strip():
            errors.append(f"{step_path}.id must be a non-empty string")
        elif step_id in seen_ids:
            errors.append(f"Duplicate step id: {step_id}")
        else:
            seen_ids.add(step_id)

        if not isinstance(tool, str) or not tool.strip():
            errors.append(f"{step_path}.tool must be a non-empty string")

        if not isinstance(arguments, dict):
            errors.append(f"{step_path}.arguments must be a dictionary")
            continue

        tool_input = arguments.get("input")

        if not isinstance(tool_input, str) or not tool_input.strip():
            errors.append(
                f"{step_path}.arguments.input "
                "must be a non-empty string"
            )

    return errors


def resolve_step_references(
    value: str,
    results: dict[str, Any],
) -> str:
    """Replace {{step_id.result}} references with prior results."""

    def replace_reference(match: re.Match[str]) -> str:
        step_id = match.group(1)

        if step_id not in results:
            raise ValueError(
                f"Unknown or unavailable step reference: {step_id}"
            )

        return str(results[step_id])

    return REFERENCE_PATTERN.sub(replace_reference, value)


def execute_tool_workflow(
    steps: list[dict[str, Any]],
    *,
    max_steps: int = 10,
    tool_caller: ToolCaller = call_tool,
) -> dict[str, Any]:
    """Execute validated tool steps in order and return results and trace."""

    validation_errors = validate_workflow_steps(
        steps,
        max_steps=max_steps,
    )

    if validation_errors:
        return {
            "status": "invalid",
            "results": {},
            "trace": [],
            "errors": validation_errors,
        }

    results: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    for index, step in enumerate(steps):
        step_id = step["id"]
        tool = step["tool"].strip().lower()
        raw_input = step["arguments"]["input"]

        trace_entry: dict[str, Any] = {
            "index": index,
            "id": step_id,
            "tool": tool,
            "raw_input": raw_input,
            "resolved_input": None,
            "status": "pending",
            "result": None,
            "error": None,
        }

        try:
            resolved_input = resolve_step_references(
                raw_input,
                results,
            )
            trace_entry["resolved_input"] = resolved_input

            result = tool_caller(tool, resolved_input)

            if isinstance(result, str) and (
                result.startswith("Unknown tool:")
                or result.startswith("Calculator error:")
                or result == "Invalid expression."
            ):
                raise RuntimeError(result)

            results[step_id] = result
            trace_entry["result"] = result
            trace_entry["status"] = "completed"

        except Exception as error:
            trace_entry["status"] = "failed"
            trace_entry["error"] = str(error)
            trace.append(trace_entry)

            return {
                "status": "failed",
                "failed_step": step_id,
                "results": results,
                "trace": trace,
                "errors": [str(error)],
            }

        trace.append(trace_entry)

    return {
        "status": "completed",
        "results": results,
        "trace": trace,
        "errors": [],
    }
