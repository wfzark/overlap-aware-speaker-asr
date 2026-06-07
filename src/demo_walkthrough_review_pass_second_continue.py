from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .demo_walkthrough_review_pass import (
    build_review_lines,
    build_review_receipt_lines,
    build_review_receipt_rows,
    build_review_row,
    load_walkthrough_steps,
)


CONTINUE_COLUMNS = [
    "review_order",
    "step_id",
    "completed_step_count",
    "focus",
    "continue_note",
]

COMPLETED_REVIEW_PATHS = (
    "results/tables/demo_walkthrough_review_pass.json",
    "results/tables/demo_walkthrough_review_pass_second.json",
    "results/tables/demo_walkthrough_review_pass_third.json",
)


def load_completed_step_ids() -> set[str]:
    completed: set[str] = set()
    for rel_path in COMPLETED_REVIEW_PATHS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            step_id = str(payload.get("step_id", "")).strip()
            if step_id:
                completed.add(step_id)
    return completed


def select_next_step(steps: list[dict[str, str]], completed_step_ids: set[str]) -> dict[str, str]:
    for step in steps:
        step_id = str(step.get("step_id", ""))
        if step_id and step_id not in completed_step_ids:
            return step
    return steps[3] if len(steps) > 3 else {"step_id": "4", "focus": "Frontier breadth"}


def build_continue_row(next_step: dict[str, str], completed_count: int) -> dict[str, str]:
    step_id = str(next_step.get("step_id", "4"))
    focus = str(next_step.get("focus", "Frontier breadth"))
    return {
        "review_order": "4",
        "step_id": step_id,
        "completed_step_count": str(completed_count),
        "focus": focus,
        "continue_note": (
            f"Walkthrough review continued to step {step_id}; no live demo or recording is claimed."
        ),
    }


def build_continue_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Walkthrough Review Pass Second Continue",
        "",
        "This generated note records the fourth qualitative walkthrough review pass in step order. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| review_order | step_id | completed_step_count | focus | continue_note |",
        "| --- | --- | ---: | --- | --- |",
        (
            f"| {row['review_order']} | {row['step_id']} | {row['completed_step_count']} | {row['focus']} | "
            f"{row['continue_note']} |"
        ),
    ]
    return lines


def write_outputs(
    continue_row: dict[str, str],
    review_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    continue_csv_path = tables_dir / "demo_walkthrough_review_pass_second_continue.csv"
    continue_json_path = tables_dir / "demo_walkthrough_review_pass_second_continue.json"
    continue_md_path = figures_dir / "demo_walkthrough_review_pass_second_continue.md"
    fourth_csv_path = tables_dir / "demo_walkthrough_review_pass_fourth.csv"
    fourth_json_path = tables_dir / "demo_walkthrough_review_pass_fourth.json"
    fourth_md_path = figures_dir / "demo_walkthrough_review_pass_fourth.md"
    receipt_json_path = tables_dir / "demo_walkthrough_review_pass_second_continue_receipt.json"
    receipt_md_path = figures_dir / "demo_walkthrough_review_pass_second_continue_receipt.md"

    with continue_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CONTINUE_COLUMNS)
        writer.writeheader()
        writer.writerow(continue_row)
    continue_json_path.write_text(json.dumps(continue_row, ensure_ascii=False, indent=2), encoding="utf-8")
    continue_md_path.write_text("\n".join(build_continue_lines(continue_row)) + "\n", encoding="utf-8")

    fourth_review_row = dict(review_row)
    fourth_review_row["review_order"] = "4"
    with fourth_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fourth_review_row.keys()))
        writer.writeheader()
        writer.writerow(fourth_review_row)
    fourth_json_path.write_text(json.dumps(fourth_review_row, ensure_ascii=False, indent=2), encoding="utf-8")
    fourth_md_path.write_text("\n".join(build_review_lines(fourth_review_row)) + "\n", encoding="utf-8")

    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_review_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return (
        continue_csv_path,
        continue_json_path,
        continue_md_path,
        fourth_csv_path,
        fourth_json_path,
        fourth_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def main() -> None:
    steps = load_walkthrough_steps()
    completed_step_ids = load_completed_step_ids()
    next_step = select_next_step(steps, completed_step_ids)
    review_row = build_review_row(next_step)
    continue_row = build_continue_row(next_step, len(completed_step_ids))
    receipt_rows = build_review_receipt_rows(review_row, len(steps))
    for receipt in receipt_rows:
        receipt["execution_status"] = "review_complete"
        receipt["review_scope"] = "fourth_walkthrough_step"
        receipt["writeback_note"] = (
            f"Fourth qualitative walkthrough review documented for step {review_row['step_id']}; "
            "live demo or recording delivery remains pending."
        )
    (
        continue_csv_path,
        continue_json_path,
        continue_md_path,
        fourth_csv_path,
        fourth_json_path,
        fourth_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(continue_row, review_row, receipt_rows)
    print(f"Wrote demo walkthrough review pass second continue CSV: {continue_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second continue JSON: {continue_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second continue note: {continue_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass fourth CSV: {fourth_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass fourth JSON: {fourth_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass fourth note: {fourth_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second continue receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second continue receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
