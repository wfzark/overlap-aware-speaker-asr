from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


RUNBOOK_COLUMNS = [
    "recommended_frontier",
    "recommended_action",
    "required_evidence",
    "completion_signal",
    "urgency",
    "runbook_note",
]


def load_operator_brief() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_operator_brief.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_handoff_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_handoff.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_runbook_card_row(
    operator_brief: dict[str, str],
    handoff_rows: list[dict[str, str]],
) -> dict[str, str]:
    if not operator_brief:
        return {}
    frontier = str(operator_brief.get("operator_frontier", ""))
    current_handoff = next((row for row in handoff_rows if row.get("frontier_name") == frontier), {})
    target_receipt = str(current_handoff.get("expected_outputs", ""))
    return {
        "recommended_frontier": frontier,
        "recommended_action": str(operator_brief.get("operator_action", "")),
        "required_evidence": str(operator_brief.get("operator_evidence", "")),
        "completion_signal": (
            f"receipt queue verification is complete and the target receipt {target_receipt} is ready to update"
            if target_receipt
            else "receipt queue verification is complete and the target receipt is ready to update"
        ),
        "urgency": str(operator_brief.get("operator_urgency", "")),
        "runbook_note": (
            f"Start with {frontier} as the current first receipt-queue target after confirming the handoff and bridge layers. "
            "This remains coordination-only and does not claim benchmark execution."
        ),
    }


def build_runbook_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Frontier Execution Receipt Queue Runbook Card",
        "",
        "This generated runbook card condenses the first receipt-queue action into a one-page execution card. "
        "It remains experimental/frontier coordination only and does not claim benchmark execution.",
        "",
        f"- Recommended frontier: `{row['recommended_frontier']}`",
        f"- Recommended action: `{row['recommended_action']}`",
        f"- Required evidence: `{row['required_evidence']}`",
        f"- Completion signal: `{row['completion_signal']}`",
        f"- Urgency: {row['urgency']}",
        f"- Runbook note: {row['runbook_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_runbook_card.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_runbook_card.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_runbook_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RUNBOOK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_runbook_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_runbook_card_row(load_operator_brief(), load_handoff_rows())
    if not row:
        print("Receipt queue operator brief not found; runbook card not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote frontier execution receipt queue runbook card CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue runbook card JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue runbook card note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
