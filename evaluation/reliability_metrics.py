from dataclasses import dataclass
from typing import Iterable


@dataclass
class ReliabilityMetrics:
    total_runs: int
    recovery_runs: int
    recovered_runs: int
    escalated_runs: int
    recovery_rate: float
    recovery_success_rate: float
    escalation_rate: float
    average_confidence_before: float
    average_confidence_after: float
    average_confidence_improvement: float


def calculate_reliability_metrics(
    reliability_records: Iterable[dict],
) -> ReliabilityMetrics:
    records = list(reliability_records)

    total_runs = len(records)

    if total_runs == 0:
        return ReliabilityMetrics(
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

    recovery_records = [
        record
        for record in records
        if record.get("recovery_attempts", 0) > 0
    ]

    recovered_records = [
        record
        for record in recovery_records
        if record.get("recovered") is True
    ]

    escalated_records = [
        record
        for record in records
        if record.get("escalated") is True
    ]

    confidence_pairs = [
        (
            record.get("confidence_before_recovery"),
            record.get("confidence_after_recovery"),
        )
        for record in records
        if record.get("confidence_before_recovery") is not None
        and record.get("confidence_after_recovery") is not None
    ]

    recovery_runs = len(recovery_records)
    recovered_runs = len(recovered_records)
    escalated_runs = len(escalated_records)

    recovery_rate = recovery_runs / total_runs

    recovery_success_rate = (
        recovered_runs / recovery_runs
        if recovery_runs > 0
        else 0.0
    )

    escalation_rate = escalated_runs / total_runs

    if confidence_pairs:
        average_confidence_before = sum(
            before for before, _ in confidence_pairs
        ) / len(confidence_pairs)

        average_confidence_after = sum(
            after for _, after in confidence_pairs
        ) / len(confidence_pairs)
    else:
        average_confidence_before = 0.0
        average_confidence_after = 0.0

    average_confidence_improvement = (
        average_confidence_after
        - average_confidence_before
    )

    return ReliabilityMetrics(
        total_runs=total_runs,
        recovery_runs=recovery_runs,
        recovered_runs=recovered_runs,
        escalated_runs=escalated_runs,
        recovery_rate=recovery_rate,
        recovery_success_rate=recovery_success_rate,
        escalation_rate=escalation_rate,
        average_confidence_before=average_confidence_before,
        average_confidence_after=average_confidence_after,
        average_confidence_improvement=average_confidence_improvement,
    )