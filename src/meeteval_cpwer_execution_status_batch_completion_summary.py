from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


COMPLETION_COLUMNS = [
    "scope",
    "ready_chain_count",
    "total_chain_count",
    "pending_chain_count",
    "queue_status",
    "observation",
]


def load_status_batch_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_execution_status_batch.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_completion_summary_row(status_rows: list[dict[str, str]]) -> dict[str, str]:
    total_count = len(status_rows)
    satisfied_statuses = {"execution_chain_ready", "execution_chain_complete"}
    ready_count = sum(1 for row in status_rows if row.get("execution_chain_status") in satisfied_statuses)
    complete_count = sum(1 for row in status_rows if row.get("execution_chain_status") == "execution_chain_complete")
    pending_count = total_count - ready_count
    if total_count == 0:
        queue_status = "queue_not_started"
    elif complete_count == total_count:
        queue_status = "queue_complete"
    elif pending_count == 0:
        queue_status = "queue_ready_to_execute"
    else:
        queue_status = "queue_in_progress"
    return {
        "scope": "meeteval_cpwer_execution_chain_batch",
        "ready_chain_count": str(ready_count),
        "total_chain_count": str(total_count),
        "pending_chain_count": str(pending_count),
        "queue_status": queue_status,
        "observation": (
            "Experimental/frontier MeetEval cpWER batch execution-chain completion rollup; "
            "official MeetEval evaluation is not claimed."
        ),
    }


def build_completion_summary_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# MeetEval cpWER Execution Status Batch Completion Summary",
        "",
        "This generated note summarizes batch execution-chain completion across all verified gold cases. "
        "It does not claim official MeetEval cpWER evaluation or benchmark completion.",
        "",
        "| scope | ready_chain_count | total_chain_count | pending_chain_count | queue_status | observation |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['ready_chain_count']} | {row['total_chain_count']} | "
            f"{row['pending_chain_count']} | {row['queue_status']} | {row['observation']} |"
        ),
    ]
    return lines


def write_outputs(completion_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_execution_status_batch_completion_summary.csv"
    json_path = tables_dir / "meeteval_cpwer_execution_status_batch_completion_summary.json"
    md_path = figures_dir / "meeteval_cpwer_execution_status_batch_completion_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_completion_summary_lines(completion_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    completion_row = build_completion_summary_row(load_status_batch_rows())
    csv_path, json_path, md_path = write_outputs(completion_row)
    print(
        "Wrote MeetEval cpWER execution status batch completion summary CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER execution status batch completion summary JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER execution status batch completion summary note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Queue status: {completion_row['queue_status']}")


if __name__ == "__main__":
    main()
