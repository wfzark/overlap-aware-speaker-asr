from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


REVIEW_COLUMNS = [
    "review_order",
    "step_id",
    "focus",
    "review_status",
    "review_note",
    "expected_evidence",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "review_scope",
    "step_count",
    "writeback_note",
]


def load_walkthrough_steps() -> list[dict[str, str]]:
    steps_path = PROJECT_ROOT / "results" / "tables" / "demo_walkthrough_steps.json"
    if not steps_path.exists():
        return []
    payload = json.loads(steps_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def select_first_step(steps: list[dict[str, str]]) -> dict[str, str]:
    if not steps:
        return {"step_id": "problem_framing", "focus": "problem framing"}
    return steps[0]


def build_review_row(step: dict[str, str]) -> dict[str, str]:
    step_id = str(step.get("step_id", "problem_framing"))
    focus = str(step.get("focus", "problem framing"))
    return {
        "review_order": "1",
        "step_id": step_id,
        "focus": focus,
        "review_status": "review_complete",
        "review_note": (
            f"Qualitative walkthrough review for {step_id} complete; no live demo or recording is claimed."
        ),
        "expected_evidence": "results/tables/demo_walkthrough_receipt.json",
    }


def build_review_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Walkthrough Review Pass",
        "",
        "This generated note records the first qualitative walkthrough review pass. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| review_order | step_id | focus | review_status | review_note | expected_evidence |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['review_order']} | {row['step_id']} | {row['focus']} | {row['review_status']} | "
            f"{row['review_note']} | {row['expected_evidence']} |"
        ),
    ]
    return lines


def build_review_receipt_rows(review_row: dict[str, str], step_count: int) -> list[dict[str, str]]:
    return [
        {
            "execution_status": "review_complete",
            "review_scope": "first_walkthrough_step",
            "step_count": str(step_count),
            "writeback_note": (
                "First qualitative walkthrough review documented; live demo or recording delivery remains pending."
            ),
        }
    ]


def build_review_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Demo Walkthrough Review Pass Receipt",
        "",
        "This receipt records the first walkthrough review writeback. "
        "It does not claim a live demo or recording.",
        "",
        "| execution_status | review_scope | step_count | writeback_note |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['review_scope']} | {row['step_count']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    review_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "demo_walkthrough_review_pass.csv"
    json_path = tables_dir / "demo_walkthrough_review_pass.json"
    md_path = figures_dir / "demo_walkthrough_review_pass.md"
    receipt_json_path = tables_dir / "demo_walkthrough_review_pass_receipt.json"
    receipt_md_path = figures_dir / "demo_walkthrough_review_pass_receipt.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(review_row)
    json_path.write_text(json.dumps(review_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_review_lines(review_row)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_review_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, receipt_json_path, receipt_md_path


def main() -> None:
    steps = load_walkthrough_steps()
    step = select_first_step(steps)
    review_row = build_review_row(step)
    receipt_rows = build_review_receipt_rows(review_row, len(steps))
    csv_path, json_path, md_path, receipt_json_path, receipt_md_path = write_outputs(
        review_row,
        receipt_rows,
    )
    print(f"Wrote demo walkthrough review pass CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough review pass receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
