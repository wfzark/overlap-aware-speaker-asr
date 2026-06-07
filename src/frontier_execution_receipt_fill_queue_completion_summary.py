from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


COMPLETION_COLUMNS = [
    "scope",
    "awaiting_fill_count",
    "total_frontier_count",
    "fill_complete_count",
    "combined_fill_status",
    "observation",
]


def load_fill_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_fill_queue_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_completion_summary_row(summary: dict[str, str]) -> dict[str, str]:
    awaiting = str(summary.get("awaiting_fill_count", "0"))
    total = str(summary.get("total_frontier_count", "0"))
    complete = str(summary.get("fill_complete_count", "0"))
    combined = str(summary.get("combined_fill_status", "fill_queue_empty"))
    return {
        "scope": "frontier_execution_receipt_fill_coordination_queue",
        "awaiting_fill_count": awaiting,
        "total_frontier_count": total,
        "fill_complete_count": complete,
        "combined_fill_status": combined,
        "observation": (
            "Experimental/frontier receipt-fill coordination queue completion rollup; "
            "template-only receipts remain unfilled and no benchmark execution is claimed."
        ),
    }


def build_completion_summary_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Fill Queue Completion Summary",
        "",
        "This generated note summarizes frontier receipt-fill coordination queue completion. "
        "It does not claim benchmark execution.",
        "",
        "| scope | awaiting_fill_count | total_frontier_count | fill_complete_count | combined_fill_status | observation |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['awaiting_fill_count']} | {row['total_frontier_count']} | "
            f"{row['fill_complete_count']} | {row['combined_fill_status']} | {row['observation']} |"
        ),
    ]
    return lines


def write_outputs(completion_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_fill_queue_completion_summary.csv"
    json_path = tables_dir / "frontier_execution_receipt_fill_queue_completion_summary.json"
    md_path = figures_dir / "frontier_execution_receipt_fill_queue_completion_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_completion_summary_lines(completion_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    completion_row = build_completion_summary_row(load_fill_summary())
    csv_path, json_path, md_path = write_outputs(completion_row)
    print(
        "Wrote frontier execution receipt fill queue completion summary CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt fill queue completion summary JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt fill queue completion summary note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Combined fill status: {completion_row['combined_fill_status']}")


if __name__ == "__main__":
    main()
