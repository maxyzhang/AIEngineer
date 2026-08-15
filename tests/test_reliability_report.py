import json

from evaluation.reliability_metrics import ReliabilityMetrics 
from evaluation.reliability_report import (
    format_reliability_report,
    load_reliability_records,
    generate_reliability_report,
)


def test_format_reliability_report():
    metrics = ReliabilityMetrics(
        total_runs=100,
        recovery_runs=20,
        recovered_runs=16,
        escalated_runs=5,
        recovery_rate=0.20,
        recovery_success_rate=0.80,
        escalation_rate=0.05,
        average_confidence_before=0.50,
        average_confidence_after=0.80,
        average_confidence_improvement=0.30,
    )

    report = format_reliability_report(metrics)

    assert "Agent Reliability Report" in report
    assert "Total Runs:" in report
    assert "100" in report
    assert "20.0%" in report
    assert "80.0%" in report
    assert "5.0%" in report
    assert "0.50" in report
    assert "0.80" in report
    assert "+0.30" in report

def test_format_empty_reliability_report():
    metrics = ReliabilityMetrics(
        total_runs=0,
        recovery_runs=0,
        recovered_runs=0,
        escalated_runs=0,
        recovery_rate=0.0,
        recovery_success_rate=0.0,
        escalation_rate=0.0,
        average_confidence_before=0.0,
        average_confidence_after=0.0,
        average_confidence_improvement=0.0,
    )

    report = format_reliability_report(metrics)

    assert "Total Runs:" in report
    assert "0.0%" in report
    assert "0.00" in report
    assert "+0.00" in report

def test_load_reliability_records(tmp_path):
    trace_file = tmp_path / "workflow_trace.jsonl"

    traces = [
        {
            "trace_id": "trace_1",
            "reliability": {
                "recovery_attempts": 1,
                "recovered": True,
                "escalated": False,
                "confidence_before_recovery": 0.40,
                "confidence_after_recovery": 0.80,
            },
        },
        {
            "trace_id": "trace_2",
            "reliability": {
                "recovery_attempts": 0,
                "recovered": False,
                "escalated": True,
                "confidence_before_recovery": 0.50,
                "confidence_after_recovery": 0.50,
            },
        },
    ]

    with trace_file.open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace) + "\n")

    records = load_reliability_records(str(trace_file))

    assert len(records) == 2
    assert records[0]["recovered"] is True
    assert records[1]["escalated"] is True 

def test_generate_reliability_report(tmp_path):
    trace_file = tmp_path / "workflow_trace.jsonl"

    trace = {
        "trace_id": "trace_1",
        "reliability": {
            "recovery_attempts": 1,
            "recovered": True,
            "escalated": False,
            "confidence_before_recovery": 0.40,
            "confidence_after_recovery": 0.80,
        },
    }

    trace_file.write_text(
        json.dumps(trace) + "\n",
        encoding="utf-8",
    )

    report = generate_reliability_report(str(trace_file))

    assert "Agent Reliability Report" in report
    assert "Total Runs:" in report
    assert "Recovered Runs:" in report
    assert "Recovery Rate:" in report
    assert "100.0%" in report
    assert "Avg Confidence Before:" in report
    assert "0.40" in report
    assert "Avg Confidence After:" in report
    assert "0.80" in report
    assert "Avg Confidence Improvement:" in report 
    assert "+0.40" in report