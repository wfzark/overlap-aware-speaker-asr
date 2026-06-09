from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


OPERATOR_BRIEF_COLUMNS = [
    "operator_frontier",
    "operator_receipt",
    "operator_evidence",
    "operator_action",
    "operator_note",
]


def load_bridge_checklist_rows() -> list[dict[str, str]]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "meeteval_tokenization_gain_frontier_fill_execution_receipt_bridge_checklist.json"
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_operator_brief_row(checklist_row: dict[str, str]) -> dict[str, str]:
    if not checklist_row:
        return {}
    frontier = str(checklist_row.get("recommended_frontier", "meeteval_compatibility"))
    receipt_target = str(
        checklist_row.get("receipt_target", "results/tables/meeteval_cpwer_execution_receipt.json")
    )
    return {
        "operator_frontier": frontier,
        "operator_receipt": receipt_target,
        "operator_evidence": (
            "results/figures/meeteval_tokenization_gain_frontier_fill_execution_receipt_bridge.md; "
            "results/figures/meeteval_tokenization_gain_frontier_fill_execution_receipt_bridge_checklist.md"
        ),
        "operator_action": (
            f"Run character-spaced cpWER for {frontier}, then write execution_status back to {receipt_target} "
            "with real evidence. Do not claim full MeetEval benchmark completion until the receipt is filled."
        ),
        "operator_note": (
            f"Tokenization gain frontier fill coordination for {frontier} is ready for receipt update. "
            f"{checklist_row.get('bridge_note', '')} This remains experimental/frontier only."
        ).strip(),
    }


def build_operator_brief_lines(row: dict[str, str]) -> list[str]:
    return [
        "# MeetEval Tokenization Gain Frontier Fill Operator Brief",
        "",
        "This generated brief gives the next contributor a plain-language MeetEval receipt fill action. "
        "It remains experimental/frontier coordination only and does not claim full benchmark completion.",
        "",
        f"- Operator frontier: `{row['operator_frontier']}`",
        f"- Operator receipt: `{row['operator_receipt']}`",
        f"- Evidence path: `{row['operator_evidence']}`",
        f"- Operator action: {row['operator_action']}",
        f"- Operator note: {row['operator_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_tokenization_gain_frontier_fill_operator_brief.csv"
    json_path = tables_dir / "meeteval_tokenization_gain_frontier_fill_operator_brief.json"
    md_path = figures_dir / "meeteval_tokenization_gain_frontier_fill_operator_brief.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OPERATOR_BRIEF_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_operator_brief_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    checklist_rows = load_bridge_checklist_rows()
    row = build_operator_brief_row(checklist_rows[0] if checklist_rows else {})
    if not row:
        print("Tokenization gain execution receipt bridge checklist not found; operator brief not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(
        "Wrote MeetEval tokenization gain frontier fill operator brief CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval tokenization gain frontier fill operator brief JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval tokenization gain frontier fill operator brief note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
