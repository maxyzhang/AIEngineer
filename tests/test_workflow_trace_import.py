from workflow_trace import (
    start_workflow_trace,
    record_reliability_trace,
    save_workflow_trace,
)


def test_reliability_trace_defaults():
    trace = start_workflow_trace(
        question="What MongoDB version are we using?",
        workflow={"name": "test"},
    )

    reliability = trace["reliability"]

    assert reliability["confidence_before_recovery"] is None
    assert reliability["confidence_after_recovery"] is None
    assert reliability["recovery_attempts"] == 0
    assert reliability["recovered"] is None
    assert reliability["recovery_stop_reason"] is None
    assert reliability["escalated"] is False
    assert reliability["escalation_severity"] is "none"
    assert reliability["escalation_reason"] is None


def test_record_reliability_trace():
    trace = start_workflow_trace(
        question="What MongoDB version are we using?",
        workflow={"name": "test"},
    )

    record_reliability_trace(
        trace,
        confidence_before_recovery=0.40,
        confidence_after_recovery=0.82,
        recovery_attempts=1,
        recovered=True,
        recovery_stop_reason="confidence_threshold_met",
        escalated=False,
        escalation_severity="none",
        escalation_reason="confidence_acceptable",
    )

    reliability = trace["reliability"]

    assert reliability["confidence_before_recovery"] == 0.40
    assert reliability["confidence_after_recovery"] == 0.82
    assert reliability["recovery_attempts"] == 1
    assert reliability["recovered"] is True
    assert reliability["recovery_stop_reason"] == "confidence_threshold_met"
    assert reliability["escalated"] is False
    assert reliability["escalation_severity"] == "none"
    assert reliability["escalation_reason"] == "confidence_acceptable"

def test_reliability_trace_can_be_saved(tmp_path, monkeypatch):
    import workflow_trace

    trace_file = tmp_path / "workflow_trace.jsonl"
    monkeypatch.setattr(workflow_trace, "TRACE_FILE", str(trace_file))

    trace = start_workflow_trace(
        question="What MongoDB version are we using?",
        workflow={"name": "test"},
    )

    record_reliability_trace(
        trace,
        confidence_before_recovery=0.40,
        confidence_after_recovery=0.82,
        recovery_attempts=1,
        recovered=True,
        recovery_stop_reason="confidence_threshold_met",
        escalated=False,
        escalation_severity="none",
        escalation_reason="confidence_acceptable",
    )

    save_workflow_trace(trace)

    content = trace_file.read_text(encoding="utf-8")

    assert "confidence_before_recovery" in content
    assert "confidence_after_recovery" in content
    assert "recovery_attempts" in content
    assert "escalated" in content
    assert "escalation_reason" in content 