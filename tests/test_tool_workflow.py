from typing import Any

from workflows.tool_workflow import (
    execute_tool_workflow,
    resolve_step_references,
    validate_workflow_steps,
)


def test_validate_workflow_rejects_non_list() -> None:
    errors = validate_workflow_steps("not a list")

    assert errors == ["Workflow steps must be a list"]


def test_validate_workflow_rejects_empty_steps() -> None:
    errors = validate_workflow_steps([])

    assert errors == ["Workflow must contain at least one step"]


def test_validate_workflow_rejects_duplicate_step_ids() -> None:
    steps = [
        {
            "id": "calculate",
            "tool": "calculator",
            "arguments": {"input": "2 + 2"},
        },
        {
            "id": "calculate",
            "tool": "calculator",
            "arguments": {"input": "4 * 2"},
        },
    ]

    errors = validate_workflow_steps(steps)

    assert "Duplicate step id: calculate" in errors


def test_validate_workflow_enforces_max_steps() -> None:
    steps = [
        {
            "id": f"step_{index}",
            "tool": "calculator",
            "arguments": {"input": "1 + 1"},
        }
        for index in range(3)
    ]

    errors = validate_workflow_steps(steps, max_steps=2)

    assert (
        "Workflow contains 3 steps, but the maximum is 2"
        in errors
    )


def test_resolve_step_references_uses_prior_result() -> None:
    resolved = resolve_step_references(
        "10 + {{first.result}}",
        {"first": "5"},
    )

    assert resolved == "10 + 5"


def test_resolve_step_references_rejects_unknown_step() -> None:
    try:
        resolve_step_references(
            "{{missing.result}} * 2",
            {},
        )
    except ValueError as error:
        assert str(error) == (
            "Unknown or unavailable step reference: missing"
        )
    else:
        raise AssertionError("Expected ValueError")


def test_execute_multi_step_calculator_workflow() -> None:
    calls: list[tuple[str, str]] = []

    def fake_tool_caller(tool: str, tool_input: str) -> str:
        calls.append((tool, tool_input))

        results = {
            "126 * 0.18": "22.68",
            "126 + 22.68": "148.68",
            "148.68 / 4": "37.17",
        }

        return results[tool_input]

    steps = [
        {
            "id": "tip",
            "tool": "calculator",
            "arguments": {
                "input": "126 * 0.18",
            },
        },
        {
            "id": "total",
            "tool": "calculator",
            "arguments": {
                "input": "126 + {{tip.result}}",
            },
        },
        {
            "id": "per_person",
            "tool": "calculator",
            "arguments": {
                "input": "{{total.result}} / 4",
            },
        },
    ]

    execution = execute_tool_workflow(
        steps,
        tool_caller=fake_tool_caller,
    )

    assert execution["status"] == "completed"
    assert execution["results"] == {
        "tip": "22.68",
        "total": "148.68",
        "per_person": "37.17",
    }

    assert calls == [
        ("calculator", "126 * 0.18"),
        ("calculator", "126 + 22.68"),
        ("calculator", "148.68 / 4"),
    ]


def test_execution_trace_records_completed_steps() -> None:
    def fake_tool_caller(tool: str, tool_input: str) -> str:
        return "4"

    steps = [
        {
            "id": "sum",
            "tool": "calculator",
            "arguments": {"input": "2 + 2"},
        }
    ]

    execution = execute_tool_workflow(
        steps,
        tool_caller=fake_tool_caller,
    )

    trace = execution["trace"]

    assert len(trace) == 1
    assert trace[0]["id"] == "sum"
    assert trace[0]["tool"] == "calculator"
    assert trace[0]["raw_input"] == "2 + 2"
    assert trace[0]["resolved_input"] == "2 + 2"
    assert trace[0]["status"] == "completed"
    assert trace[0]["result"] == "4"
    assert trace[0]["error"] is None


def test_workflow_stops_when_tool_fails() -> None:
    calls: list[str] = []

    def fake_tool_caller(tool: str, tool_input: str) -> Any:
        calls.append(tool_input)

        if tool_input == "bad input":
            raise RuntimeError("Tool failed")

        return "should not run"

    steps = [
        {
            "id": "first",
            "tool": "calculator",
            "arguments": {"input": "bad input"},
        },
        {
            "id": "second",
            "tool": "calculator",
            "arguments": {"input": "2 + 2"},
        },
    ]

    execution = execute_tool_workflow(
        steps,
        tool_caller=fake_tool_caller,
    )

    assert execution["status"] == "failed"
    assert execution["failed_step"] == "first"
    assert execution["results"] == {}
    assert calls == ["bad input"]
    assert len(execution["trace"]) == 1
    assert execution["trace"][0]["status"] == "failed"
    assert execution["trace"][0]["error"] == "Tool failed"


def test_workflow_stops_for_unavailable_reference() -> None:
    calls: list[str] = []

    def fake_tool_caller(tool: str, tool_input: str) -> str:
        calls.append(tool_input)
        return "unused"

    steps = [
        {
            "id": "calculation",
            "tool": "calculator",
            "arguments": {
                "input": "{{missing.result}} + 1",
            },
        }
    ]

    execution = execute_tool_workflow(
        steps,
        tool_caller=fake_tool_caller,
    )

    assert execution["status"] == "failed"
    assert execution["failed_step"] == "calculation"
    assert calls == []
    assert execution["errors"] == [
        "Unknown or unavailable step reference: missing"
    ]


def test_invalid_workflow_does_not_call_tools() -> None:
    calls: list[str] = []

    def fake_tool_caller(tool: str, tool_input: str) -> str:
        calls.append(tool_input)
        return "unused"

    steps = [
        {
            "id": "",
            "tool": "calculator",
            "arguments": {"input": "2 + 2"},
        }
    ]

    execution = execute_tool_workflow(
        steps,
        tool_caller=fake_tool_caller,
    )

    assert execution["status"] == "invalid"
    assert execution["results"] == {}
    assert execution["trace"] == []
    assert calls == []

