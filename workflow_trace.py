from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


TRACE_FILE = "workflow_trace.jsonl"


def create_trace_id() -> str:
    """
    Create a readable unique trace identifier.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"workflow_{timestamp}"


def start_workflow_trace(
    *,
    question: str,
    workflow: dict[str, Any],
) -> dict[str, Any]:
    """
    Initialize tracing for a workflow execution.
    """

    return {
        "trace_id": create_trace_id(),
        "started_at": datetime.now().isoformat(),
        "question": question,
        "workflow": workflow,
        "steps": [],
        "status": "running",
        "fallback_used": False,
        "_started_perf": perf_counter(),
    }


def start_step_trace(
    *,
    trace: dict[str, Any],
    step_number: int,
    tool_name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """
    Create trace information for one workflow step.
    """

    step_trace = {
        "step_number": step_number,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "_started_perf": perf_counter(),
    }

    trace["steps"].append(step_trace)
    return step_trace


def complete_step_trace(
    *,
    step_trace: dict[str, Any],
    observation: Any,
) -> None:
    """
    Mark a workflow step as successfully completed.
    """

    started_perf = step_trace.pop(
        "_started_perf",
        perf_counter(),
    )

    step_trace.update(
        {
            "completed_at": datetime.now().isoformat(),
            "duration_ms": round(
                (perf_counter() - started_perf) * 1000,
                2,
            ),
            "status": "completed",
            "observation": observation,
        }
    )


def fail_step_trace(
    *,
    step_trace: dict[str, Any],
    error: Exception | str,
) -> None:
    """
    Mark a workflow step as failed.
    """

    started_perf = step_trace.pop(
        "_started_perf",
        perf_counter(),
    )

    step_trace.update(
        {
            "completed_at": datetime.now().isoformat(),
            "duration_ms": round(
                (perf_counter() - started_perf) * 1000,
                2,
            ),
            "status": "failed",
            "error": str(error),
        }
    )


def complete_workflow_trace(
    *,
    trace: dict[str, Any],
    final_answer: str,
) -> None:
    """
    Mark the complete workflow as successful.
    """

    started_perf = trace.pop(
        "_started_perf",
        perf_counter(),
    )

    trace.update(
        {
            "completed_at": datetime.now().isoformat(),
            "duration_ms": round(
                (perf_counter() - started_perf) * 1000,
                2,
            ),
            "status": "completed",
            "final_answer": final_answer,
        }
    )


def fail_workflow_trace(
    *,
    trace: dict[str, Any],
    error: Exception | str,
    fallback_used: bool,
) -> None:
    """
    Mark the workflow as failed.
    """

    started_perf = trace.pop(
        "_started_perf",
        perf_counter(),
    )

    trace.update(
        {
            "completed_at": datetime.now().isoformat(),
            "duration_ms": round(
                (perf_counter() - started_perf) * 1000,
                2,
            ),
            "status": "failed",
            "error": str(error),
            "fallback_used": fallback_used,
        }
    )


def save_workflow_trace(
    trace: dict[str, Any],
    *,
    trace_file: str = TRACE_FILE,
) -> bool:
    """
    Append one workflow trace as a JSON Lines record.
    """

    safe_trace = {
        key: value
        for key, value in trace.items()
        if not key.startswith("_")
    }

    try:
        path = Path(trace_file)

        with path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    safe_trace,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

        return True

    except OSError as error:
        print(
            "[Workflow Trace] "
            f"Failed to save trace: {error}"
        )
        return False
