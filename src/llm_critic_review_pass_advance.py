from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .llm_critic_review_pass import (
    build_review_pass_lines,
    build_review_pass_receipt_lines,
    build_review_pass_receipt_rows,
    build_review_pass_row,
    load_qualitative_row,
    load_review_queue,
)


ADVANCE_COLUMNS = [
    "queue_order",
    "case_id",
    "prior_pass_status",
    "review_priority",
    "review_outcome",
    "advance_note",
]


def load_completed_pass_case() -> str:
    pass_path = PROJECT_ROOT / "results" / "tables" / "llm_critic_review_pass.json"
    if not pass_path.exists():
        return ""
    payload = json.loads(pass_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return str(payload.get("case_id", ""))
    return ""


def select_next_queue_row(queue_rows: list[dict[str, str]], completed_case_id: str) -> dict[str, str]:
    for row in queue_rows:
        case_id = str(row.get("case_id", ""))
        if case_id and case_id != completed_case_id:
            return row
    return queue_rows[0] if queue_rows else {"case_id": "LightOverlap", "review_priority": "high"}


def build_advance_row(
    queue_row: dict[str, str],
    pass_row: dict[str, str],
    completed_case_id: str,
) -> dict[str, str]:
    case_id = str(pass_row.get("case_id", ""))
    return {
        "queue_order": str(queue_row.get("queue_order", "2")),
        "case_id": case_id,
        "prior_pass_status": f"{completed_case_id or 'HeavyOverlap'} review_complete",
        "review_priority": str(pass_row.get("review_priority", "high")),
        "review_outcome": str(pass_row.get("review_outcome", "")),
        "advance_note": (
            f"Queue advanced to {case_id} after the first qualitative pass; no verified transcript repair was applied."
        ),
    }


def build_advance_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# LLM Critic Review Pass Advance",
        "",
        "This generated note records the second qualitative critic-style review pass in queue order. "
        "It does not claim verified transcript correction.",
        "",
        "| queue_order | case_id | prior_pass_status | review_priority | review_outcome | advance_note |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['queue_order']} | {row['case_id']} | {row['prior_pass_status']} | {row['review_priority']} | "
            f"{row['review_outcome']} | {row['advance_note']} |"
        ),
    ]
    return lines


def write_outputs(
    advance_row: dict[str, str],
    pass_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    advance_csv_path = tables_dir / "llm_critic_review_pass_advance.csv"
    advance_json_path = tables_dir / "llm_critic_review_pass_advance.json"
    advance_md_path = figures_dir / "llm_critic_review_pass_advance.md"
    pass_csv_path = tables_dir / "llm_critic_review_pass_second.csv"
    pass_json_path = tables_dir / "llm_critic_review_pass_second.json"
    pass_md_path = figures_dir / "llm_critic_review_pass_second.md"
    receipt_json_path = tables_dir / "llm_critic_review_pass_advance_receipt.json"
    receipt_md_path = figures_dir / "llm_critic_review_pass_advance_receipt.md"

    with advance_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ADVANCE_COLUMNS)
        writer.writeheader()
        writer.writerow(advance_row)
    advance_json_path.write_text(json.dumps(advance_row, ensure_ascii=False, indent=2), encoding="utf-8")
    advance_md_path.write_text("\n".join(build_advance_lines(advance_row)) + "\n", encoding="utf-8")

    with pass_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pass_row.keys()))
        writer.writeheader()
        writer.writerow(pass_row)
    pass_json_path.write_text(json.dumps(pass_row, ensure_ascii=False, indent=2), encoding="utf-8")
    pass_md_path.write_text("\n".join(build_review_pass_lines(pass_row)) + "\n", encoding="utf-8")

    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_review_pass_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return (
        advance_csv_path,
        advance_json_path,
        advance_md_path,
        pass_csv_path,
        pass_json_path,
        pass_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def main() -> None:
    queue_rows = load_review_queue()
    completed_case_id = load_completed_pass_case()
    next_row = select_next_queue_row(queue_rows, completed_case_id)
    case_id = str(next_row.get("case_id", "LightOverlap"))
    qualitative_row = load_qualitative_row(case_id)
    pass_row = build_review_pass_row(next_row, qualitative_row)
    advance_row = build_advance_row(next_row, pass_row, completed_case_id)
    receipt_rows = build_review_pass_receipt_rows(pass_row)
    for receipt in receipt_rows:
        receipt["writeback_note"] = (
            f"Second qualitative pass complete for {case_id}; no verified repair claim was made."
        )
    (
        advance_csv_path,
        advance_json_path,
        advance_md_path,
        pass_csv_path,
        pass_json_path,
        pass_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(advance_row, pass_row, receipt_rows)
    print(f"Wrote LLM critic review pass advance CSV: {advance_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass advance JSON: {advance_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass advance note: {advance_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass second CSV: {pass_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass second JSON: {pass_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass second note: {pass_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass advance receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic review pass advance receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
