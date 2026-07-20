import json

from workflows.tool_workflow_planner import (
    build_tool_workflow_prompt,
    create_tool_workflow,
    extract_step_references,
    parse_tool_workflow_response,
    validate_supported_tools_and_references,
)


def valid_workflow_response() -> str:
    return json.dumps(
        {
            "steps": [
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
            ]
        }
    )


def test_build_prompt_includes_question_and_rules() -> None:
    prompt = build_tool_workflow_prompt(
        "Calculate a tip",
        max_steps=4,
    )

    assert "Calculate a tip" in prompt
    assert "Return between 1 and 4 steps" in prompt
    assert "{{step_id.result}}" in prompt
    assert '"steps"' in prompt


def test_extract_step_references() -> None:
    references = extract_step_references(
        "{{first.result}} + {{second.result}}"
    )

    assert references == ["first", "second"]


def test_parse_valid_workflow_response() -> None:
    parsed = parse_tool_workflow_response(
        valid_workflow_response()
    )

    assert parsed["status"] == "valid"
    assert parsed["errors"] == []
    assert len(parsed["steps"]) == 2
    assert parsed["steps"][1]["id"] == "total"


def test_parse_removes_markdown_code_fence() -> None:
    response = (
        "```json\n"
        + valid_workflow_response()
        + "\n```"
    )

    parsed = parse_tool_workflow_response(response)

    assert parsed["status"] == "valid"
    assert len(parsed["steps"]) == 2


def test_parse_rejects_invalid_json() -> None:
    parsed = parse_tool_workflow_response(
        '{"steps": [}'
    )

    assert parsed["status"] == "invalid"
    assert parsed["steps"] == []
    assert parsed["errors"][0].startswith(
        "Workflow response is not valid JSON:"
    )


def test_parse_rejects_non_object_json() -> None:
    parsed = parse_tool_workflow_response(
        '["not", "an", "object"]'
    )

    assert parsed["status"] == "invalid"
    assert parsed["errors"] == [
        "Workflow response must be a JSON object"
    ]


def test_parse_rejects_missing_steps() -> None:
    parsed = parse_tool_workflow_response(
        '{"answer": "no workflow"}'
    )

    assert parsed["status"] == "invalid"
    assert "Workflow steps must be a list" in parsed["errors"]


def test_rejects_unsupported_tool() -> None:
    steps = [
        {
            "id": "email",
            "tool": "email",
            "arguments": {
                "input": "Send a message",
            },
        }
    ]

    errors = validate_supported_tools_and_references(steps)

    assert errors == [
        "Unsupported tool in steps[0]: email"
    ]


def test_rejects_reference_to_later_step() -> None:
    steps = [
        {
            "id": "first",
            "tool": "calculator",
            "arguments": {
                "input": "{{second.result}} + 1",
            },
        },
        {
            "id": "second",
            "tool": "calculator",
            "arguments": {
                "input": "2 + 2",
            },
        },
    ]

    errors = validate_supported_tools_and_references(steps)

    assert errors == [
        "Step reference must point to an earlier step: second"
    ]


def test_accepts_reference_to_earlier_step() -> None:
    steps = [
        {
            "id": "first",
            "tool": "calculator",
            "arguments": {
                "input": "2 + 2",
            },
        },
        {
            "id": "second",
            "tool": "calculator",
            "arguments": {
                "input": "{{first.result}} * 2",
            },
        },
    ]

    errors = validate_supported_tools_and_references(steps)

    assert errors == []


def test_create_tool_workflow_uses_injected_generator() -> None:
    captured_prompts: list[str] = []

    def fake_generator(prompt: str) -> str:
        captured_prompts.append(prompt)
        return valid_workflow_response()

    workflow = create_tool_workflow(
        "Calculate an 18 percent tip",
        workflow_generator=fake_generator,
    )

    assert workflow["status"] == "valid"
    assert len(workflow["steps"]) == 2
    assert len(captured_prompts) == 1
    assert (
        "Calculate an 18 percent tip"
        in captured_prompts[0]
    )


def test_create_tool_workflow_rejects_empty_question() -> None:
    called = False

    def fake_generator(prompt: str) -> str:
        nonlocal called
        called = True
        return valid_workflow_response()

    workflow = create_tool_workflow(
        "   ",
        workflow_generator=fake_generator,
    )

    assert workflow["status"] == "invalid"
    assert workflow["errors"] == [
        "Question must be a non-empty string"
    ]
    assert called is False


def test_create_tool_workflow_handles_generator_failure() -> None:
    def failing_generator(prompt: str) -> str:
        raise RuntimeError("LLM unavailable")

    workflow = create_tool_workflow(
        "Calculate a tip",
        workflow_generator=failing_generator,
    )

    assert workflow["status"] == "failed"
    assert workflow["steps"] == []
    assert workflow["errors"] == ["LLM unavailable"]


def test_create_three_step_calculator_workflow() -> None:
    response = json.dumps(
        {
            "steps": [
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
        }
    )

    workflow = create_tool_workflow(
        "Calculate tip and divide among four people",
        workflow_generator=lambda prompt: response,
    )

    assert workflow["status"] == "valid"
    assert [
        step["id"]
        for step in workflow["steps"]
    ] == [
        "tip",
        "total",
        "per_person",
    ]


def test_parse_enforces_max_steps() -> None:
    response = json.dumps(
        {
            "steps": [
                {
                    "id": f"step_{index}",
                    "tool": "calculator",
                    "arguments": {
                        "input": "1 + 1",
                    },
                }
                for index in range(3)
            ]
        }
    )

    parsed = parse_tool_workflow_response(
        response,
        max_steps=2,
    )

    assert parsed["status"] == "invalid"
    assert (
        "Workflow contains 3 steps, but the maximum is 2"
        in parsed["errors"]
    )

