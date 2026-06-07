from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


COMPLETION_COLUMNS = [
    "scope",
    "ready_receipt_count",
    "total_receipt_count",
    "pending_receipt_count",
    "queue_status",
    "observation",
]

READINESS_KEYS = [
    "meeteval_readiness_status",
    "speaker_profile_readiness_status",
    "external_staging_readiness_status",
]


def load_status_row() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_status.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def count_ready_receipts(status_row: dict[str, str]) -> tuple[int, int]:
    total = len(READINESS_KEYS)
    ready = sum(1 for key in READINESS_KEYS if status_row.get(key) == "receipt_ready_to_fill")
    return ready, total


def build_completion_summary_row(status_row: dict[str, str]) -> dict[str, str]:
    ready_count, total_count = count_ready_receipts(status_row)
    pending_count = total_count - ready_count
    queue_status = "queue_complete" if pending_count == 0 else "queue_in_progress"
    return {
        "scope": "frontier_execution_receipt_coordination_queue",
        "ready_receipt_count": str(ready_count),
        "total_receipt_count": str(total_count),
        "pending_receipt_count": str(pending_count),
        "queue_status": queue_status,
        "observation": (
            "Experimental/frontier execution-receipt coordination queue completion rollup; "
            "no official benchmark execution or external audio staging is claimed."
        ),
    }


def build_completion_summary_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Queue Completion Summary",
        "",
        "This generated note summarizes frontier execution-receipt coordination queue completion. "
        "It does not claim benchmark execution.",
        "",
        "| scope | ready_receipt_count | total_receipt_count | pending_receipt_count | queue_status | observation |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['ready_receipt_count']} | {row['total_receipt_count']} | "
            f"{row['pending_receipt_count']} | {row['queue_status']} | {row['observation']} |"
        ),
    ]
    return lines


def write_outputs(completion_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_completion_summary.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_completion_summary.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_completion_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_completion_summary_lines(completion_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    completion_row = build_completion_summary_row(load_status_row())
    csv_path, json_path, md_path = write_outputs(completion_row)
    print(f"Wrote frontier execution receipt queue completion summary CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue completion summary JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue completion summary note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Queue status: {completion_row['queue_status']}")


if __name__ == "__main__":
    main()
