import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "memory_metrics.py"


def run_cli(
    *arguments: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """
    Run the memory metrics CLI in an isolated directory.
    """

    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def create_minimal_runtime_files(
    directory: Path,
) -> None:
    """
    Create minimal input files required by the report utility.
    """

    memory_data = {
        "last_question": "",
        "last_answer": "",
        "long_term_memory": [],
        "updated_at": "",
    }

    audit_event = {
        "timestamp": "2026-07-19T00:00:00",
        "event_type": "reinforced",
        "memory_text": "Test memory",
        "details": {
            "access_count": 1,
        },
    }

    (directory / "memory.json").write_text(
        json.dumps(memory_data),
        encoding="utf-8",
    )

    (directory / "memory_audit.jsonl").write_text(
        json.dumps(audit_event) + "\n",
        encoding="utf-8",
    )


def test_help_returns_success(
    tmp_path: Path,
) -> None:
    result = run_cli(
        "--help",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "--report-file" in result.stdout
    assert "--history-dir" in result.stdout
    assert "--history-limit" in result.stdout
    assert "--no-history" in result.stdout


def test_invalid_history_limit_returns_code_2(
    tmp_path: Path,
) -> None:
    result = run_cli(
        "--history-limit",
        "1",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert (
        "--history-limit must be at least 2."
        in result.stderr
    )


def test_no_history_creates_latest_report_only(
    tmp_path: Path,
) -> None:
    create_minimal_runtime_files(tmp_path)

    result = run_cli(
        "--no-history",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert (tmp_path / "memory_report.json").exists()
    assert not (tmp_path / "memory_reports").exists()
    assert "History snapshot: disabled" in result.stdout


def test_custom_report_file_is_created(
    tmp_path: Path,
) -> None:
    create_minimal_runtime_files(tmp_path)

    report_file = tmp_path / "custom_report.json"

    result = run_cli(
        "--report-file",
        str(report_file),
        "--no-history",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert report_file.exists()

    report = json.loads(
        report_file.read_text(encoding="utf-8")
    )

    assert "metrics" in report
    assert "health" in report
    assert "warnings" in report
    assert "recommendations" in report
    assert "trend" in report


def test_custom_history_directory_creates_snapshot(
    tmp_path: Path,
) -> None:
    create_minimal_runtime_files(tmp_path)

    history_dir = tmp_path / "custom_history"

    result = run_cli(
        "--history-dir",
        str(history_dir),
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert history_dir.exists()

    snapshot_files = list(
        history_dir.glob("memory_report_*.json")
    )

    assert len(snapshot_files) == 1


def test_report_file_cannot_be_directory(
    tmp_path: Path,
) -> None:
    invalid_report_path = tmp_path / "report_directory"
    invalid_report_path.mkdir()

    result = run_cli(
        "--report-file",
        str(invalid_report_path),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert (
        "--report-file must point to a file, "
        "not a directory."
        in result.stderr
    )


def test_history_dir_cannot_be_file(
    tmp_path: Path,
) -> None:
    invalid_history_path = tmp_path / "history_file"
    invalid_history_path.write_text(
        "test",
        encoding="utf-8",
    )

    result = run_cli(
        "--history-dir",
        str(invalid_history_path),
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert (
        "--history-dir must point to a directory."
        in result.stderr
    )
