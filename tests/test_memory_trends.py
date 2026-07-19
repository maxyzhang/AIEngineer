from typing import Any

from memory_metrics import (
    assess_memory_trend,
    compare_memory_reports,
)


def make_report(
    *,
    generated_at: str,
    total_memories: int = 5,
    average_importance: float = 5.0,
    average_access_count: float = 2.0,
    stale_memories: int = 2,
    high_value_memories: int = 2,
    never_accessed_memories: int = 1,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a minimal memory report for trend tests.
    """

    return {
        "generated_at": generated_at,
        "health": {
            "total_memories": total_memories,
            "average_importance": average_importance,
            "average_access_count": average_access_count,
            "stale_memories": stale_memories,
            "high_value_memories": high_value_memories,
            "never_accessed_memories": never_accessed_memories,
        },
        "warnings": warnings or [],
    }


def test_compare_memory_reports_requires_two_reports() -> None:
    comparison = compare_memory_reports([])

    assert comparison == {
        "available": False,
        "message": (
            "At least two historical reports are required."
        ),
    }


def test_compare_memory_reports_calculates_changes() -> None:
    latest = make_report(
        generated_at="2026-07-19T10:00:00",
        total_memories=7,
        average_importance=6.0,
        average_access_count=3.5,
        stale_memories=1,
        high_value_memories=4,
        never_accessed_memories=0,
        warnings=["warning"],
    )

    previous = make_report(
        generated_at="2026-07-18T10:00:00",
        total_memories=5,
        average_importance=5.0,
        average_access_count=2.0,
        stale_memories=3,
        high_value_memories=2,
        never_accessed_memories=2,
        warnings=["warning one", "warning two"],
    )

    comparison = compare_memory_reports(
        [latest, previous]
    )

    assert comparison["available"] is True
    assert (
        comparison["latest_generated_at"]
        == "2026-07-19T10:00:00"
    )
    assert (
        comparison["previous_generated_at"]
        == "2026-07-18T10:00:00"
    )

    assert comparison["changes"] == {
        "total_memories": 2.0,
        "average_importance": 1.0,
        "average_access_count": 1.5,
        "stale_memories": -2.0,
        "high_value_memories": 2.0,
        "never_accessed_memories": -2.0,
        "warning_count": -1,
    }


def test_compare_memory_reports_handles_invalid_values() -> None:
    latest = {
        "generated_at": "latest",
        "health": {
            "average_importance": "invalid",
            "average_access_count": None,
        },
        "warnings": "not-a-list",
    }

    previous = {
        "generated_at": "previous",
        "health": {},
        "warnings": [],
    }

    comparison = compare_memory_reports(
        [latest, previous]
    )

    changes = comparison["changes"]

    assert changes["average_importance"] == 0.0
    assert changes["average_access_count"] == 0.0

def test_assess_memory_trend_returns_insufficient_history() -> None:
    comparison = {
        "available": False,
    }

    assert (
        assess_memory_trend(comparison)
        == "Insufficient history"
    )


def test_assess_memory_trend_detects_improving() -> None:
    comparison = {
        "available": True,
        "changes": {
            "stale_memories": -1,
            "never_accessed_memories": -1,
            "average_importance": 1,
            "average_access_count": 1,
            "high_value_memories": 1,
            "warning_count": -1,
        },
    }

    assert assess_memory_trend(comparison) == "Improving"


def test_assess_memory_trend_detects_declining() -> None:
    comparison = {
        "available": True,
        "changes": {
            "stale_memories": 1,
            "never_accessed_memories": 1,
            "average_importance": -1,
            "average_access_count": -1,
            "high_value_memories": -1,
            "warning_count": 1,
        },
    }

    assert assess_memory_trend(comparison) == "Declining"


def test_assess_memory_trend_detects_stable() -> None:
    comparison = {
        "available": True,
        "changes": {
            "stale_memories": 0,
            "never_accessed_memories": 0,
            "average_importance": 0,
            "average_access_count": 0,
            "high_value_memories": 0,
            "warning_count": 0,
        },
    }

    assert assess_memory_trend(comparison) == "Stable"


def test_assess_memory_trend_balanced_score_is_stable() -> None:
    comparison = {
        "available": True,
        "changes": {
            "stale_memories": -1,
            "never_accessed_memories": 1,
            "average_importance": 1,
            "average_access_count": -1,
            "high_value_memories": 0,
            "warning_count": 0,
        },
    }

    assert assess_memory_trend(comparison) == "Stable"
