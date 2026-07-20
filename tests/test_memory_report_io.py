import json
from pathlib import Path
from typing import Any

from memory_metrics import (
    export_memory_report,
    load_recent_memory_reports,
    validate_memory_report,
)

def make_report(
    *,
    generated_at: str = "2026-07-19T10:00:00",
    total_memories: int = 5,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "latest_audit_event": "2026-07-19T09:00:00",
        "metrics": {
            "total_events": 2,
            "unique_memories": 2,
            "event_counts": {
                "reinforced": 1,
                "consolidated": 1,
            },
        },
        "health": {
            "total_memories": total_memories,
            "average_importance": 5.0,
            "average_access_count": 2.0,
            "stale_memories": 1,
            "high_value_memories": 2,
            "never_accessed_memories": 1,
        },
        "warnings": [],
        "recommendations": [
            "No immediate action is required."
        ],
        "trend": {
            "status": "Stable",
            "comparison": {
                "available": False,
            },
        },
    }

def test_load_recent_memory_reports_returns_latest_first(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / "memory_reports"
    history_dir.mkdir()

    older = make_report(
        generated_at="2026-07-18T10:00:00",
    )
    newer = make_report(
        generated_at="2026-07-19T10:00:00",
    )

    (history_dir / "memory_report_20260718_100000.json").write_text(
        json.dumps(older),
        encoding="utf-8",
    )
    (history_dir / "memory_report_20260719_100000.json").write_text(
        json.dumps(newer),
        encoding="utf-8",
    )

    reports = load_recent_memory_reports(
        history_dir=str(history_dir),
        limit=2,
    )

    assert len(reports) == 2
    assert reports[0]["generated_at"] == "2026-07-19T10:00:00"
    assert reports[1]["generated_at"] == "2026-07-18T10:00:00"

def test_load_recent_memory_reports_respects_limit(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / "memory_reports"
    history_dir.mkdir()

    for index in range(3):
        report = make_report(
            generated_at=f"2026-07-{17 + index:02d}T10:00:00",
        )

        report_path = (
            history_dir
            / f"memory_report_202607{17 + index:02d}_100000.json"
        )

        report_path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    reports = load_recent_memory_reports(
        history_dir=str(history_dir),
        limit=2,
    )

    assert len(reports) == 2

def test_load_recent_memory_reports_handles_missing_directory(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"

    reports = load_recent_memory_reports(
        history_dir=str(missing_dir),
        limit=5,
    )

    assert reports == []

def test_load_recent_memory_reports_skips_invalid_json(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / "memory_reports"
    history_dir.mkdir()

    valid_report = make_report()

    (history_dir / "memory_report_20260719_100000.json").write_text(
        json.dumps(valid_report),
        encoding="utf-8",
    )

    (history_dir / "memory_report_20260719_110000.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    reports = load_recent_memory_reports(
        history_dir=str(history_dir),
        limit=5,
    )

    assert len(reports) == 1
    assert reports[0]["generated_at"] == "2026-07-19T10:00:00"

def test_load_recent_memory_reports_skips_non_object_json(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / "memory_reports"
    history_dir.mkdir()

    (history_dir / "memory_report_20260719_100000.json").write_text(
        json.dumps(["not", "a", "report"]),
        encoding="utf-8",
    )

    reports = load_recent_memory_reports(
        history_dir=str(history_dir),
        limit=5,
    )

    assert reports == []

def test_export_memory_report_writes_latest_report(
    tmp_path: Path,
) -> None:
    report_file = tmp_path / "memory_report.json"
    history_dir = tmp_path / "memory_reports"

    succeeded = export_memory_report(
        metrics_summary={
            "total_events": 2,
            "unique_memories": 2,
            "event_counts": {
                "reinforced": 1,
                "consolidated": 1,
            },
        },
        health_summary={
            "total_memories": 5,
            "average_importance": 5.0,
            "average_access_count": 2.0,
            "stale_memories": 1,
            "high_value_memories": 2,
            "never_accessed_memories": 1,
        },
        warnings=[],
        recommendations=[
            "No immediate action is required."
        ],
        latest_audit_timestamp="2026-07-19T09:00:00",
        trend_comparison={
            "available": False,
        },
        trend_status="Stable",
        report_file=str(report_file),
        history_dir=str(history_dir),
        save_history=False,
    )

    assert succeeded is True
    assert report_file.exists()
    assert not history_dir.exists()

    report = json.loads(
        report_file.read_text(encoding="utf-8")
    )

    assert report["metrics"]["total_events"] == 2
    assert report["health"]["total_memories"] == 5
    assert report["trend"]["status"] == "Stable"

def test_export_memory_report_writes_history_snapshot(
    tmp_path: Path,
) -> None:
    report_file = tmp_path / "memory_report.json"
    history_dir = tmp_path / "memory_reports"

    succeeded = export_memory_report(
        metrics_summary={
            "total_events": 0,
            "unique_memories": 0,
            "event_counts": {},
        },
        health_summary={
            "total_memories": 0,
            "average_importance": 0.0,
            "average_access_count": 0.0,
            "stale_memories": 0,
            "high_value_memories": 0,
            "never_accessed_memories": 0,
        },
        warnings=[
            "No persistent long-term memories are available."
        ],
        recommendations=[
            "Review memory extraction and persistence."
        ],
        latest_audit_timestamp=None,
        trend_comparison={
            "available": False,
        },
        trend_status="Insufficient history",
        report_file=str(report_file),
        history_dir=str(history_dir),
        save_history=True,
    )

    assert succeeded is True
    assert report_file.exists()
    assert history_dir.exists()

    snapshots = list(
        history_dir.glob("memory_report_*.json")
    )

    assert len(snapshots) == 1

    snapshot = json.loads(
        snapshots[0].read_text(encoding="utf-8")
    )

    assert snapshot["health"]["total_memories"] == 0
    assert snapshot["trend"]["status"] == "Insufficient history"

def valid_report() -> dict:
    return {
        "generated_at": "2026-07-19T10:00:00",
        "metrics": {
            "total_events": 2,
            "unique_memories": 2,
            "event_counts": {
                "consolidated": 1,
                "reinforced": 1,
            },
        },
        "health": {
            "total_memories": 5,
            "average_importance": 4.6,
            "average_access_count": 2.0,
            "stale_memories": 3,
            "high_value_memories": 2,
            "never_accessed_memories": 2,
        },
        "warnings": [],
        "recommendations": [],
        "trend": {"status": "stable"},
    }


def test_validate_memory_report_accepts_valid_report() -> None:
    assert validate_memory_report(valid_report()) is True


def test_validate_memory_report_rejects_non_dictionary() -> None:
    assert validate_memory_report([]) is False


def test_validate_memory_report_rejects_missing_required_field() -> None:
    report = valid_report()
    del report["health"]

    assert validate_memory_report(report) is False


def test_validate_memory_report_rejects_invalid_top_level_type() -> None:
    report = valid_report()
    report["warnings"] = "No warnings"

    assert validate_memory_report(report) is False


def test_validate_memory_report_rejects_invalid_metrics() -> None:
    report = valid_report()
    report["metrics"]["total_events"] = "2"

    assert validate_memory_report(report) is False


def test_validate_memory_report_rejects_invalid_health() -> None:
    report = valid_report()
    report["health"]["average_importance"] = "high"

    assert validate_memory_report(report) is False 

