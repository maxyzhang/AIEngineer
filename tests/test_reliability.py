import pytest

from evaluation.reliability_metrics import calculate_reliability_metrics


def test_empty_reliability_records():
    metrics = calculate_reliability_metrics([])

    assert metrics.total_runs == 0
    assert metrics.recovery_runs == 0
    assert metrics.recovered_runs == 0
    assert metrics.escalated_runs == 0
    assert metrics.recovery_rate == 0.0
    assert metrics.recovery_success_rate == 0.0
    assert metrics.escalation_rate == 0.0
    assert metrics.average_confidence_before == 0.0
    assert metrics.average_confidence_after == 0.0
    assert metrics.average_confidence_improvement == 0.0


def test_recovery_and_escalation_rates():
    records = [
        {
            "recovery_attempts": 1,
            "recovered": True,
            "escalated": False,
            "confidence_before_recovery": 0.40,
            "confidence_after_recovery": 0.80,
        },
        {
            "recovery_attempts": 2,
            "recovered": False,
            "escalated": True,
            "confidence_before_recovery": 0.30,
            "confidence_after_recovery": 0.50,
        },
        {
            "recovery_attempts": 0,
            "recovered": None,
            "escalated": False,
            "confidence_before_recovery": None,
            "confidence_after_recovery": None,
        },
    ]

    metrics = calculate_reliability_metrics(records)

    assert metrics.total_runs == 3
    assert metrics.recovery_runs == 2
    assert metrics.recovered_runs == 1
    assert metrics.escalated_runs == 1

    assert metrics.recovery_rate == pytest.approx(2 / 3)
    assert metrics.recovery_success_rate == pytest.approx(0.5)
    assert metrics.escalation_rate == pytest.approx(1 / 3)


def test_average_confidence_improvement():
    records = [
        {
            "recovery_attempts": 1,
            "recovered": True,
            "escalated": False,
            "confidence_before_recovery": 0.40,
            "confidence_after_recovery": 0.80,
        },
        {
            "recovery_attempts": 1,
            "recovered": True,
            "escalated": False,
            "confidence_before_recovery": 0.60,
            "confidence_after_recovery": 0.90,
        },
    ]

    metrics = calculate_reliability_metrics(records)

    assert metrics.average_confidence_before == pytest.approx(0.50)
    assert metrics.average_confidence_after == pytest.approx(0.85)
    assert metrics.average_confidence_improvement == pytest.approx(0.35)


def test_missing_confidence_values_are_ignored():
    records = [
        {
            "recovery_attempts": 0,
            "recovered": None,
            "escalated": False,
            "confidence_before_recovery": None,
            "confidence_after_recovery": None,
        },
        {
            "recovery_attempts": 1,
            "recovered": True,
            "escalated": False,
            "confidence_before_recovery": 0.50,
            "confidence_after_recovery": 0.75,
        },
    ]

    metrics = calculate_reliability_metrics(records)

    assert metrics.total_runs == 2
    assert metrics.average_confidence_before == pytest.approx(0.50)
    assert metrics.average_confidence_after == pytest.approx(0.75)
    assert metrics.average_confidence_improvement == pytest.approx(0.25) 