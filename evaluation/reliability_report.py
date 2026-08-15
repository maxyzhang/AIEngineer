import json
from pathlib import Path
from evaluation.reliability_metrics import (
    ReliabilityMetrics,
    calculate_reliability_metrics,
)

def format_reliability_report(
    metrics: ReliabilityMetrics,
) -> str:
    return f"""
=== Agent Reliability Report ===

Total Runs:                 {metrics.total_runs}
Recovery Runs:              {metrics.recovery_runs}
Recovered Runs:             {metrics.recovered_runs}
Escalated Runs:             {metrics.escalated_runs}

Recovery Rate:              {metrics.recovery_rate:.1%}
Recovery Success Rate:      {metrics.recovery_success_rate:.1%}
Escalation Rate:            {metrics.escalation_rate:.1%}

Avg Confidence Before:      {metrics.average_confidence_before:.2f}
Avg Confidence After:       {metrics.average_confidence_after:.2f}
Avg Confidence Improvement: {metrics.average_confidence_improvement:+.2f}
""".strip()

def load_reliability_records(
        trace_file: str = "workflow_trace.jsonl",
) -> list[dict]:
    path = Path(trace_file)

    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue
            trace = json.loads(line)

            reliability = trace.get("reliability")
            if reliability is not None:
                records.append(reliability)

    return records

def generate_reliability_report(
        trace_file: str = "workflow_trace.jsonl",
) -> str:
    records = load_reliability_records(trace_file)

    metrics = calculate_reliability_metrics(records)

    return format_reliability_report(metrics)