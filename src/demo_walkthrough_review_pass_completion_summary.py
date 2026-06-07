from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .demo_walkthrough_review_pass import load_walkthrough_steps
from .demo_walkthrough_review_pass_status import (
    build_status_lines,
    build_status_row,
    load_completed_step_ids,
)


COMPLETION_COLUMNS = [
    "scope",
    "completed_count",
    "total_step_count",
    "pending_count",
    "queue_status",
    "observation",
]


def build_completion_summary_row(status_row: dict[str, str]) -> dict[str, str]:
    pending_count = int(status_row.get("pending_count", 0) or 0)
    queue_status = "queue_complete" if pending_count == 0 else "queue_in_progress"
    return {
        "scope": "walkthrough_review_queue",
        "completed_count": str(status_row.get("completed_count", "0")),
        "total_step_count": str(status_row.get("total_step_count", "0")),
        "pending_count": str(status_row.get("pending_count", "0")),
        "queue_status": queue_status,
        "observation": (
            "Qualitative/demo walkthrough review queue completion rollup; "
            "no live demo or recording delivery is claimed."
        ),
    }


def build_completion_summary_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Walkthrough Review Pass Completion Summary",
        "",
        "This generated note summarizes walkthrough-queue completion for the qualitative demo pass layer. "
        "It does not claim live demo or recording delivery.",
        "",
        "| scope | completed_count | total_step_count | pending_count | queue_status | observation |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['completed_count']} | {row['total_step_count']} | "
            f"{row['pending_count']} | {row['queue_status']} | {row['observation']} |"
        ),
    ]
    return lines


def write_outputs(
    status_row: dict[str, str],
    completion_row: dict[str, str],
) -> tuple[Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    status_csv_path = tables_dir / "demo_walkthrough_review_pass_status.csv"
    status_json_path = tables_dir / "demo_walkthrough_review_pass_status.json"
    status_md_path = figures_dir / "demo_walkthrough_review_pass_status.md"
    completion_csv_path = tables_dir / "demo_walkthrough_review_pass_completion_summary.csv"
    completion_json_path = tables_dir / "demo_walkthrough_review_pass_completion_summary.json"
    completion_md_path = figures_dir / "demo_walkthrough_review_pass_completion_summary.md"

    with status_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(status_row.keys()))
        writer.writeheader()
        writer.writerow(status_row)
    status_json_path.write_text(json.dumps(status_row, ensure_ascii=False, indent=2), encoding="utf-8")
    status_md_path.write_text("\n".join(build_status_lines(status_row)) + "\n", encoding="utf-8")

    with completion_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    completion_json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    completion_md_path.write_text("\n".join(build_completion_summary_lines(completion_row)) + "\n", encoding="utf-8")
    return (
        status_csv_path,
        status_json_path,
        status_md_path,
        completion_csv_path,
        completion_json_path,
        completion_md_path,
    )


def main() -> None:
    steps = load_walkthrough_steps()
    completed_step_ids = load_completed_step_ids()
    status_row = build_status_row(steps, completed_step_ids)
    completion_row = build_completion_summary_row(status_row)
    (
        status_csv_path,
        status_json_path,
        status_md_path,
        completion_csv_path,
        completion_json_path,
        completion_md_path,
    ) = write_outputs(status_row, completion_row)
    print(f"Wrote demo walkthrough review pass status CSV: {status_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass status JSON: {status_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass status note: {status_md_path.relative_to(PROJECT_ROOT)}")
    print(
        f"Wrote demo walkthrough review pass completion summary CSV: {completion_csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Wrote demo walkthrough review pass completion summary JSON: {completion_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Wrote demo walkthrough review pass completion summary note: {completion_md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Queue status: {completion_row['queue_status']}")


if __name__ == "__main__":
    main()
