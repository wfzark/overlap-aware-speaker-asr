from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "step_id",
    "review_status",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_third_review_row() -> dict[str, str]:
    review_path = PROJECT_ROOT / "results" / "tables" / "demo_walkthrough_review_pass_third.json"
    if not review_path.exists():
        return {}
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(review_row: dict[str, str]) -> list[dict[str, str]]:
    step_id = str(review_row.get("step_id", "3"))
    review_status = str(review_row.get("review_status", "review_complete"))
    focus = str(review_row.get("focus", "Routing takeaway"))
    return [
        {
            "checklist_order": "1",
            "step_id": step_id,
            "review_status": review_status,
            "prerequisite_artifact": "results/figures/demo_walkthrough_review_pass_third.md",
            "receipt_target": "results/figures/demo_walkthrough_review_pass_second_continue.md",
            "checklist_goal": (
                f"Verify the third walkthrough review bridge before advancing to step 4 ({focus})."
            ),
            "bridge_note": (
                f"Step {step_id} remains {review_status}; confirm routing takeaway review before frontier breadth pass."
            ),
            "next_gate": "Confirm this bridge before opening the demo walkthrough second continue target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Demo Walkthrough Review Pass Continue Bridge Checklist",
        "",
        "This generated checklist turns the third walkthrough review into a row-by-row bridge verification path. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| checklist_order | step_id | review_status | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['step_id']} | {row['review_status']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "demo_walkthrough_review_pass_continue_bridge_checklist.csv"
    json_path = tables_dir / "demo_walkthrough_review_pass_continue_bridge_checklist.json"
    md_path = figures_dir / "demo_walkthrough_review_pass_continue_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    review_row = load_third_review_row()
    rows = build_bridge_checklist_rows(review_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote demo walkthrough review pass continue bridge checklist CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass continue bridge checklist JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass continue bridge checklist note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
