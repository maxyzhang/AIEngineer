import json
import os
from collections import Counter
from typing import Any


AUDIT_LOG_FILE = "memory_audit.jsonl"


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
    summary = summarize_memory_events(events)
    print_memory_metrics(summary)


if __name__ == "__main__":
    main()
