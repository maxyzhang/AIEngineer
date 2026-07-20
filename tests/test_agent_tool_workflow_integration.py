from typing import Any

import agent_loop


def completed_execution() -> dict[str, Any]:
    return {
        "status": "completed",
        "results": {
            "tip": "22.68",
            "total": "148.68",
        },
        "trace": [
            {
                "index": 0,
                "id": "tip",
                "tool": "calculator",
                "raw_input": "126 * 0.18",
                "resolved_input": "126 * 0.18",
                "status": "completed",
                "result": "22.68",
                "error": None,
            },
            {
                "index": 1,
                "id": "total",
                "tool": "calculator",
                "raw_input": "126 + {{tip.result}}",
                "resolved_input": "126 + 22.68",
                "status": "completed",
                "result": "148.68",
                "error": None,
            },
        ],
        "errors": [],
    }


def test_format_workflow_observation() -> None:
    observation = agent_loop.format_workflow_observation(
        completed_execution()
    )

    assert "Structured Tool Workflow Results:" in observation
    assert "Status: completed" in observation
    assert "Step: tip" in observation
    assert "Tool: calculator" in observation
    assert "Input: 126 * 0.18" in observation
    assert "Result: 22.68" in observation
    assert "Step: total" in observation
    assert "Result: 148.68" in observation


def test_format_workflow_observation_includes_failure() -> None:
    execution = {
        "status": "failed",
        "results": {},
        "trace": [
            {
                "id": "lookup",
                "tool": "search",
                "resolved_input": "current price",
                "status": "failed",
                "result": None,
                "error": "Search unavailable",
            }
        ],
        "errors": ["Search unavailable"],
    }

    observation = agent_loop.format_workflow_observation(
        execution
    )

    assert "Status: failed" in observation
    assert "Error: Search unavailable" in observation
    assert "Workflow Errors:" in observation
    assert "- Search unavailable" in observation


def test_run_structured_tool_workflow_executes_valid_plan(
    monkeypatch,
) -> None:
    plan = {
        "status": "valid",
        "steps": [
            {
                "id": "sum",
                "tool": "calculator",
                "arguments": {
                    "input": "2 + 2",
                },
            }
        ],
        "errors": [],
        "raw_response": "{}",
    }

    execution = {
        "status": "completed",
        "results": {"sum": "4"},
        "trace": [
            {
                "id": "sum",
                "tool": "calculator",
                "resolved_input": "2 + 2",
                "status": "completed",
                "result": "4",
                "error": None,
            }
        ],
        "errors": [],
    }

    monkeypatch.setattr(
        agent_loop,
        "create_tool_workflow",
        lambda question, max_steps: plan,
    )
    monkeypatch.setattr(
        agent_loop,
        "execute_tool_workflow",
        lambda steps, max_steps: execution,
    )

    result = agent_loop.run_structured_tool_workflow(
        "What is 2 plus 2?",
        max_steps=4,
    )

    assert result["status"] == "completed"
    assert result["plan"] == plan
    assert result["execution"] == execution
    assert "Result: 4" in result["observation"]


def test_run_structured_tool_workflow_handles_planning_failure(
    monkeypatch,
) -> None:
    failed_plan = {
        "status": "invalid",
        "steps": [],
        "errors": ["Invalid workflow"],
        "raw_response": "",
    }

    monkeypatch.setattr(
        agent_loop,
        "create_tool_workflow",
        lambda question, max_steps: failed_plan,
    )

    result = agent_loop.run_structured_tool_workflow(
        "Do something",
    )

    assert result["status"] == "planning_failed"
    assert result["plan"] == failed_plan
    assert result["execution"] is None
    assert result["observation"] == ""


def test_try_structured_tool_workflow_generates_answer(
    monkeypatch,
) -> None:
    workflow_result = {
        "status": "completed",
        "plan": {},
        "execution": completed_execution(),
        "observation": "workflow evidence",
    }

    captured: dict[str, str] = {}

    monkeypatch.setattr(
        agent_loop,
        "run_structured_tool_workflow",
        lambda question, max_steps: workflow_result,
    )

    def fake_generate_final_answer(
        question: str,
        history: str,
        memory_context: str = "",
        conversation_context: str = "",
    ) -> str:
        captured["question"] = question
        captured["history"] = history
        captured["memory_context"] = memory_context
        captured["conversation_context"] = conversation_context
        return "Final workflow answer"

    monkeypatch.setattr(
        agent_loop,
        "generate_final_answer",
        fake_generate_final_answer,
    )

    answer = agent_loop.try_structured_tool_workflow(
        "Calculate the dinner total",
        max_steps=5,
    )

    assert answer == "Final workflow answer"
    assert captured == {
        "question": "Calculate the dinner total",
        "history": "workflow evidence",
        "memory_context": "",
        "conversation_context": "",
    }


def test_try_structured_tool_workflow_returns_none_on_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent_loop,
        "run_structured_tool_workflow",
        lambda question, max_steps: {
            "status": "failed",
            "plan": {},
            "execution": {},
            "observation": "failed workflow",
        },
    )

    answer = agent_loop.try_structured_tool_workflow(
        "Calculate something",
    )

    assert answer is None
