from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "queue_status",
    "completed_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_completion_summary() -> dict[str, str]:
    summary_path = PROJECT_ROOT / "results" / "tables" / "demo_storyboard_review_pass_completion_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    queue_status = str(summary.get("queue_status", "queue_in_progress"))
    completed_count = str(summary.get("completed_count", "0"))
    return [
        {
            "checklist_order": "1",
            "queue_status": queue_status,
            "completed_count": completed_count,
            "prerequisite_artifact": "results/figures/demo_storyboard_review_pass_completion_summary.md",
            "receipt_target": "results/figures/demo_excellence_queue_status.md",
            "checklist_goal": (
                "Verify the storyboard completion summary bridge before opening the demo excellence queue status."
            ),
            "bridge_note": (
                f"Storyboard completion reports queue_status={queue_status} with completed_count={completed_count}; "
                "confirm storyboard queue closure before advancing the unified demo excellence rollup."
            ),
            "next_gate": "Confirm this bridge before opening the demo excellence queue status target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Demo Storyboard Review Pass Completion Summary Bridge Checklist",
        "",
        "This generated checklist turns the storyboard completion summary into a row-by-row bridge verification path. "
        "It remains qualitative/demo only and does not claim live demo or recording delivery.",
        "",
        "| checklist_order | queue_status | completed_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['queue_status']} | {row['completed_count']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "demo_storyboard_review_pass_completion_summary_bridge_checklist.csv"
    json_path = tables_dir / "demo_storyboard_review_pass_completion_summary_bridge_checklist.json"
    md_path = figures_dir / "demo_storyboard_review_pass_completion_summary_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_completion_summary()
    rows = build_bridge_checklist_rows(summary)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote demo storyboard review pass completion summary bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote demo storyboard review pass completion summary bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote demo storyboard review pass completion summary bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
