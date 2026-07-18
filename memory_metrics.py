import json
import os
from collections import Counter
from typing import Any
from datetime import datetime
from pathlib import Path


AUDIT_LOG_FILE = "memory_audit.jsonl"
MEMORY_FILE = "memory.json"
REPORT_FILE = "memory_report.json"
REPORT_HISTORY_DIR = "memory_reports"

def load_recent_memory_reports(
    history_dir: str = REPORT_HISTORY_DIR,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """
    Load the most recent valid historical memory reports.
    """

    history_path = Path(history_dir)

    if not history_path.exists():
        return []

    report_files = sorted(
        history_path.glob("memory_report_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    reports: list[dict[str, Any]] = []

    for report_file in report_files:
        try:
            with report_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                report = json.load(file)

            if isinstance(report, dict):
                reports.append(report)

        except (OSError, json.JSONDecodeError) as error:
            print(
                "[Memory Trends] "
                f"Skipped invalid report {report_file.name}: {error}"
            )
            continue

        if len(reports) >= limit:
            break

    return reports

def compare_memory_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare the two most recent memory reports.
    """

    if len(reports) < 2:
        return {
            "available": False,
            "message": "At least two historical reports are required.",
        }

    latest = reports[0]
    previous = reports[1]

    latest_health = latest.get("health", {})
    previous_health = previous.get("health", {})

    latest_warnings = latest.get("warnings", [])
    previous_warnings = previous.get("warnings", [])

    def number(
        source: dict[str, Any],
        key: str,
    ) -> float:
        try:
            return float(source.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    return {
        "available": True,
        "latest_generated_at": latest.get(
            "generated_at",
            "Unknown",
        ),
        "previous_generated_at": previous.get(
            "generated_at",
            "Unknown",
        ),
        "changes": {
            "total_memories": (
                number(latest_health, "total_memories")
                - number(previous_health, "total_memories")
            ),
            "average_importance": round(
                number(latest_health, "average_importance")
                - number(previous_health, "average_importance"),
                2,
            ),
            "average_access_count": round(
                number(latest_health, "average_access_count")
                - number(previous_health, "average_access_count"),
                2,
            ),
            "stale_memories": (
                number(latest_health, "stale_memories")
                - number(previous_health, "stale_memories")
            ),
            "high_value_memories": (
                number(latest_health, "high_value_memories")
                - number(previous_health, "high_value_memories")
            ),
            "never_accessed_memories": (
                number(latest_health, "never_accessed_memories")
                - number(previous_health, "never_accessed_memories")
            ),
            "warning_count": (
                len(latest_warnings)
                - len(previous_warnings)
            ),
        },
    }

def assess_memory_trend(
    comparison: dict[str, Any],
) -> str:
    """
    Classify the overall memory health trend.
    """

    if not comparison.get("available"):
        return "Insufficient history"

    changes = comparison.get("changes", {})

    score = 0

    if changes.get("stale_memories", 0) < 0:
        score += 1
    elif changes.get("stale_memories", 0) > 0:
        score -= 1

    if changes.get("never_accessed_memories", 0) < 0:
        score += 1
    elif changes.get("never_accessed_memories", 0) > 0:
        score -= 1

    if changes.get("average_importance", 0) > 0:
        score += 1
    elif changes.get("average_importance", 0) < 0:
        score -= 1

    if changes.get("average_access_count", 0) > 0:
        score += 1
    elif changes.get("average_access_count", 0) < 0:
        score -= 1

    if changes.get("high_value_memories", 0) > 0:
        score += 1
    elif changes.get("high_value_memories", 0) < 0:
        score -= 1

    if changes.get("warning_count", 0) < 0:
        score += 1
    elif changes.get("warning_count", 0) > 0:
        score -= 1

    if score > 0:
        return "Improving"

    if score < 0:
        return "Declining"

    return "Stable"

def print_memory_trend(
    comparison: dict[str, Any],
    trend_status: str,
) -> None:
    """
    Print the comparison between the two latest reports.
    """

    print("\n[Memory Trend]")
    print("=" * 50)

    if not comparison.get("available"):
        print(comparison.get("message"))
        print("=" * 50)
        return

    changes = comparison.get("changes", {})

    print(
        "Previous report: "
        f"{comparison.get('previous_generated_at')}"
    )
    print(
        "Latest report: "
        f"{comparison.get('latest_generated_at')}"
    )
    print("-" * 50)

    print(
        f"Total memories change: "
        f"{changes.get('total_memories', 0):+g}"
    )
    print(
        f"Average importance change: "
        f"{changes.get('average_importance', 0):+g}"
    )
    print(
        f"Average access count change: "
        f"{changes.get('average_access_count', 0):+g}"
    )
    print(
        f"Stale memories change: "
        f"{changes.get('stale_memories', 0):+g}"
    )
    print(
        f"High-value memories change: "
        f"{changes.get('high_value_memories', 0):+g}"
    )
    print(
        f"Never-accessed memories change: "
        f"{changes.get('never_accessed_memories', 0):+g}"
    )
    print(
        f"Warnings change: "
        f"{changes.get('warning_count', 0):+g}"
    )

    print("-" * 50)
    print(f"Overall trend: {trend_status}")
    print("=" * 50)

def load_audit_events(
    audit_file: str = AUDIT_LOG_FILE,
) -> list[dict[str, Any]]:
    """
    Load valid JSON Lines audit events.

    Invalid or empty lines are skipped so one damaged record does not
    prevent the remaining metrics from being generated.
    """

    if not os.path.exists(audit_file):
        return []

    events: list[dict[str, Any]] = []

    try:
        with open(audit_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        "[Memory Metrics] "
                        f"Skipped invalid JSON at line {line_number}"
                    )
                    continue

                if isinstance(event, dict):
                    events.append(event)

    except OSError as error:
        print(
            "[Memory Metrics] Failed to read audit log:",
            error,
        )

    return events

def load_memory_data(
    memory_file: str = MEMORY_FILE,
) -> dict[str, Any]:
    """
    Load persistent memory data safely.
    """

    if not os.path.exists(memory_file):
        return {
            "long_term_memory": [],
        }

    try:
        with open(memory_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError) as error:
        print(
            "[Memory Health] Failed to load memory file:",
            error,
        )

    return {
        "long_term_memory": [],
    }

def summarize_memory_health(
    memory: dict[str, Any],
    stale_days: int = 90,
    high_value_importance: int = 8,
) -> dict[str, Any]:
    """
    Summarize the current health of persistent long-term memory.
    """

    now = datetime.now()

    items = memory.get("long_term_memory", [])

    valid_items = [
        item
        for item in items
        if isinstance(item, dict)
        and str(item.get("text", "")).strip()
    ]

    if not valid_items:
        return {
            "total_memories": 0,
            "average_importance": 0.0,
            "average_access_count": 0.0,
            "stale_memories": 0,
            "high_value_memories": 0,
            "never_accessed_memories": 0,
        }

    importance_values = []
    access_counts = []

    stale_count = 0
    high_value_count = 0
    never_accessed_count = 0

    for item in valid_items:
        importance = int(item.get("importance", 5))
        access_count = int(item.get("access_count", 0))

        importance_values.append(importance)
        access_counts.append(access_count)

        if importance >= high_value_importance:
            high_value_count += 1

        last_accessed = item.get("last_accessed")

        if not last_accessed:
            never_accessed_count += 1

        reference_timestamp = (
            last_accessed
            or item.get("created_at")
        )

        if not reference_timestamp:
            continue

        try:
            reference_time = datetime.fromisoformat(
                reference_timestamp
            )
        except (TypeError, ValueError):
            continue

        age_days = max(
            (now - reference_time).total_seconds() / 86400.0,
            0.0,
        )

        if age_days >= stale_days:
            stale_count += 1

    return {
        "total_memories": len(valid_items),
        "average_importance": round(
            sum(importance_values) / len(importance_values),
            2,
        ),
        "average_access_count": round(
            sum(access_counts) / len(access_counts),
            2,
        ),
        "stale_memories": stale_count,
        "high_value_memories": high_value_count,
        "never_accessed_memories": never_accessed_count,
    }

def generate_memory_health_warnings(
    health: dict[str, Any],
    stale_ratio_threshold: float = 0.40,
    never_accessed_ratio_threshold: float = 0.40,
    minimum_average_importance: float = 4.0,
    minimum_high_value_memories: int = 1,
) -> list[str]:
    """
    Generate warnings based on memory health indicators.
    """

    warnings: list[str] = []

    total_memories = int(
        health.get("total_memories", 0)
    )

    if total_memories <= 0:
        warnings.append(
            "No persistent long-term memories are available."
        )
        return warnings

    stale_memories = int(
        health.get("stale_memories", 0)
    )

    never_accessed_memories = int(
        health.get("never_accessed_memories", 0)
    )

    high_value_memories = int(
        health.get("high_value_memories", 0)
    )

    average_importance = float(
        health.get("average_importance", 0.0)
    )

    stale_ratio = stale_memories / total_memories
    never_accessed_ratio = (
        never_accessed_memories / total_memories
    )

    if stale_ratio >= stale_ratio_threshold:
        warnings.append(
            "A high percentage of memories are stale: "
            f"{stale_memories}/{total_memories} "
            f"({stale_ratio:.0%})."
        )

    if (
        never_accessed_ratio
        >= never_accessed_ratio_threshold
    ):
        warnings.append(
            "A high percentage of memories have never been "
            f"accessed: {never_accessed_memories}/"
            f"{total_memories} "
            f"({never_accessed_ratio:.0%})."
        )

    if average_importance < minimum_average_importance:
        warnings.append(
            "Average memory importance is low: "
            f"{average_importance:.2f}."
        )

    if high_value_memories < minimum_high_value_memories:
        warnings.append(
            "No high-value memories are currently available."
        )

    return warnings

def generate_memory_recommendations(
    health: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    """
    Generate recommended actions from memory health indicators.
    """

    recommendations: list[str] = []

    total_memories = int(
        health.get("total_memories", 0)
    )

    stale_memories = int(
        health.get("stale_memories", 0)
    )

    never_accessed_memories = int(
        health.get("never_accessed_memories", 0)
    )

    high_value_memories = int(
        health.get("high_value_memories", 0)
    )

    average_importance = float(
        health.get("average_importance", 0.0)
    )

    if total_memories <= 0:
        recommendations.append(
            "Add durable user facts before evaluating memory health."
        )
        return recommendations

    if stale_memories > 0:
        recommendations.append(
            "Review stale memories and run decay or garbage "
            "collection where appropriate."
        )

    if never_accessed_memories > 0:
        recommendations.append(
            "Review memory extraction quality and remove facts "
            "that are too vague or unlikely to be retrieved."
        )

    if average_importance < 4.0:
        recommendations.append(
            "Review importance scoring rules and verify that "
            "durable facts receive appropriate initial scores."
        )

    if high_value_memories <= 0:
        recommendations.append(
            "Review reinforcement thresholds so frequently useful "
            "memories can become high-value."
        )

    if not warnings:
        recommendations.append(
            "No immediate action is required. Continue monitoring "
            "memory lifecycle metrics."
        )

    return recommendations

def export_memory_report(
    metrics_summary: dict[str, Any],
    health_summary: dict[str, Any],
    warnings: list[str],
    recommendations: list[str],
       latest_audit_timestamp: str,
    trend_comparison: dict[str, any],
    trend_status: str,
    report_file: str = REPORT_FILE,
) -> bool:
    """
    Export the complete memory observability report as JSON.
    """

    report = {
        "generated_at": datetime.now().isoformat(),
        "latest_audit_event": latest_audit_timestamp,
        "metrics": metrics_summary,
        "health": health_summary,
        "warnings": warnings,
        "recommendations": recommendations,
        "trend": {
            "status": trend_status,
            "comparison": trend_comparison,
        },
    }

    try:
        with open(
            report_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        snapshot_file = save_memory_report_snapshot(report)
        print(
            "\n[Memory Report Export]"
        )
        print("=" * 50)
        print(f"Latest report : {report_file}")

        if snapshot_file:
            print(f"History snapshot: {snapshot_file}")
            
        print("=" * 50)

        return True

    except OSError as error:
        print(
            "[Memory Report Export] Failed to write report:",
            error,
        )
        return False

def save_memory_report_snapshot(
    report: dict[str, Any],
    history_dir: str = REPORT_HISTORY_DIR,
) -> str | None:
    """
    Save a timestamped copy of the memory report.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_path = Path(history_dir)
    snapshot_file = history_path / f"memory_report_{timestamp}.json"

    try:
        history_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        with snapshot_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return str(snapshot_file)

    except OSError as error:
        print(
            "[Memory Report History] "
            f"Failed to write snapshot: {error}"
        )
        return None

def print_memory_recommendations(
    recommendations: list[str],
) -> None:
    """
    Print recommended memory maintenance actions.
    """

    print("\n[Memory Recommendations]")
    print("=" * 50)

    if not recommendations:
        print("No recommendations available.")
    else:
        for recommendation in recommendations:
            print(f"- {recommendation}")

    print("=" * 50)

def print_memory_health_warnings(
    warnings: list[str],
) -> None:
    """
    Print memory health warnings.
    """

    print("\n[Memory Health Warnings]")
    print("=" * 50)

    if not warnings:
        print("No health warnings detected.")
    else:
        for warning in warnings:
            print(f"- {warning}")

    print("=" * 50)

def get_latest_audit_timestamp(
    events: list[dict[str, Any]],
) -> str:
    """
    Return the most recent valid audit timestamp.
    """

    valid_times = []

    for event in events:
        timestamp = event.get("timestamp")

        if not timestamp:
            continue

        try:
            valid_times.append(
                datetime.fromisoformat(str(timestamp))
            )
        except (TypeError, ValueError):
            continue

    if not valid_times:
        return "No audit events"

    latest = max(valid_times)

    return latest.isoformat()

def print_memory_health(
    health: dict[str, Any],
    latest_audit_timestamp: str,
) -> None:
    """
    Print a readable memory health report.
    """

    print("\n[Memory Health]")
    print("=" * 50)

    print(
        f"Total memories: "
        f"{health.get('total_memories', 0)}"
    )

    print(
        f"Average importance: "
        f"{health.get('average_importance', 0.0)}"
    )

    print(
        f"Average access count: "
        f"{health.get('average_access_count', 0.0)}"
    )

    print(
        f"High-value memories: "
        f"{health.get('high_value_memories', 0)}"
    )

    print(
        f"Stale memories: "
        f"{health.get('stale_memories', 0)}"
    )

    print(
        f"Never-accessed memories: "
        f"{health.get('never_accessed_memories', 0)}"
    )

    print(
        f"Latest audit event: "
        f"{latest_audit_timestamp}"
    )

    print("=" * 50)

def summarize_memory_events(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a summary of memory lifecycle activity.
    """

    event_counts = Counter(
        str(event.get("event_type", "unknown"))
        for event in events
    )

    unique_memories = {
        str(event.get("memory_text", "")).strip()
        for event in events
        if str(event.get("memory_text", "")).strip()
    }

    return {
        "total_events": len(events),
        "unique_memories": len(unique_memories),
        "event_counts": dict(event_counts),
    }


def print_memory_metrics(
    summary: dict[str, Any],
) -> None:
    """
    Print a readable command-line summary.
    """

    event_counts = summary.get("event_counts", {})

    print("\n[Memory Metrics]")
    print("=" * 50)
    print(f"Total events: {summary.get('total_events', 0)}")
    print(
        "Unique memories affected: "
        f"{summary.get('unique_memories', 0)}"
    )
    print("-" * 50)

    event_types = [
        "reinforced",
        "importance_increased",
        "decayed",
        "consolidated",
        "garbage_collected",
    ]

    for event_type in event_types:
        print(
            f"{event_type}: "
            f"{event_counts.get(event_type, 0)}"
        )

    unknown_count = event_counts.get("unknown", 0)

    if unknown_count:
        print(f"unknown: {unknown_count}")

    print("=" * 50)


def main() -> None:
    events = load_audit_events()
    memory = load_memory_data()

    metrics_summary = summarize_memory_events(events)
    health_summary = summarize_memory_health(memory)

    latest_audit_timestamp = get_latest_audit_timestamp(
        events
    )

    warnings = generate_memory_health_warnings(
        health_summary
    )

    recommendations = generate_memory_recommendations(
        health_summary,
        warnings,
    )

    recent_reports = load_recent_memory_reports()
    trend_comparison = compare_memory_reports(
        recent_reports
    )
    trend_status = assess_memory_trend(
        trend_comparison
    )

    print_memory_metrics(metrics_summary)

    print_memory_health(
        health_summary,
        latest_audit_timestamp,
    )

    print_memory_health_warnings(warnings)
    print_memory_recommendations(recommendations)
    print_memory_trend(
        trend_comparison,
        trend_status,
    )

    export_memory_report(
        metrics_summary,
        health_summary,
        warnings,
        recommendations,
        latest_audit_timestamp,
        trend_comparison,
        trend_status,
    )

if __name__ == "__main__":
    main()
