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


ADVANCE_COLUMNS = [
    "review_order",
    "step_id",
    "prior_step_status",
    "focus",
    "advance_note",
]


def load_completed_review_step() -> str:
    review_path = PROJECT_ROOT / "results" / "tables" / "demo_walkthrough_review_pass.json"
    if not review_path.exists():
        return ""
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return str(payload.get("step_id", ""))
    return ""


def select_next_step(steps: list[dict[str, str]], completed_step_id: str) -> dict[str, str]:
    for index, step in enumerate(steps):
        step_id = str(step.get("step_id", ""))
        if step_id and step_id != completed_step_id:
            return step
        if index == 0 and not completed_step_id:
            return step
    return steps[1] if len(steps) > 1 else {"step_id": "2", "focus": "Baseline evidence"}


def build_advance_row(next_step: dict[str, str], completed_step_id: str) -> dict[str, str]:
    step_id = str(next_step.get("step_id", "2"))
    focus = str(next_step.get("focus", "Baseline evidence"))
    return {
        "review_order": "2",
        "step_id": step_id,
        "prior_step_status": f"{completed_step_id or '1'} review_complete",
        "focus": focus,
        "advance_note": (
            f"Walkthrough review advanced to step {step_id}; no live demo or recording is claimed."
        ),
    }


def build_advance_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Walkthrough Review Pass Advance",
        "",
        "This generated note records the second qualitative walkthrough review pass in step order. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| review_order | step_id | prior_step_status | focus | advance_note |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| {row['review_order']} | {row['step_id']} | {row['prior_step_status']} | {row['focus']} | "
            f"{row['advance_note']} |"
        ),
    ]
    return lines


def write_outputs(
    advance_row: dict[str, str],
    review_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    advance_csv_path = tables_dir / "demo_walkthrough_review_pass_advance.csv"
    advance_json_path = tables_dir / "demo_walkthrough_review_pass_advance.json"
    advance_md_path = figures_dir / "demo_walkthrough_review_pass_advance.md"
    second_csv_path = tables_dir / "demo_walkthrough_review_pass_second.csv"
    second_json_path = tables_dir / "demo_walkthrough_review_pass_second.json"
    second_md_path = figures_dir / "demo_walkthrough_review_pass_second.md"
    receipt_json_path = tables_dir / "demo_walkthrough_review_pass_advance_receipt.json"
    receipt_md_path = figures_dir / "demo_walkthrough_review_pass_advance_receipt.md"

    with advance_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ADVANCE_COLUMNS)
        writer.writeheader()
        writer.writerow(advance_row)
    advance_json_path.write_text(json.dumps(advance_row, ensure_ascii=False, indent=2), encoding="utf-8")
    advance_md_path.write_text("\n".join(build_advance_lines(advance_row)) + "\n", encoding="utf-8")

    second_review_row = dict(review_row)
    second_review_row["review_order"] = "2"
    with second_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(second_review_row.keys()))
        writer.writeheader()
        writer.writerow(second_review_row)
    second_json_path.write_text(json.dumps(second_review_row, ensure_ascii=False, indent=2), encoding="utf-8")
    second_md_path.write_text("\n".join(build_review_lines(second_review_row)) + "\n", encoding="utf-8")

    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_review_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return (
        advance_csv_path,
        advance_json_path,
        advance_md_path,
        second_csv_path,
        second_json_path,
        second_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def main() -> None:
    steps = load_walkthrough_steps()
    completed_step_id = load_completed_review_step()
    next_step = select_next_step(steps, completed_step_id)
    review_row = build_review_row(next_step)
    advance_row = build_advance_row(next_step, completed_step_id)
    receipt_rows = build_review_receipt_rows(review_row, len(steps))
    for receipt in receipt_rows:
        receipt["execution_status"] = "review_complete"
        receipt["review_scope"] = "second_walkthrough_step"
        receipt["writeback_note"] = (
            f"Second qualitative walkthrough review documented for step {review_row['step_id']}; "
            "live demo or recording delivery remains pending."
        )
    (
        advance_csv_path,
        advance_json_path,
        advance_md_path,
        second_csv_path,
        second_json_path,
        second_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(advance_row, review_row, receipt_rows)
    print(f"Wrote demo walkthrough review pass advance CSV: {advance_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass advance JSON: {advance_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass advance note: {advance_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second CSV: {second_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second JSON: {second_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass second note: {second_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass advance receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass advance receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
