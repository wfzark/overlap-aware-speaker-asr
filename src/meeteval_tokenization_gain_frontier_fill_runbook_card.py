from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


RUNBOOK_COLUMNS = [
    "runbook_status",
    "recommended_frontier",
    "adapted_case_ratio",
    "handoff_goal",
    "next_action",
    "required_evidence",
    "completion_signal",
    "guardrail_note",
]


def load_tokenization_handoff() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_tokenization_gain_to_frontier_fill_handoff.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_frontier_runbook_card() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_fill_execution_runbook_card.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_runbook_card_row(
    tokenization_handoff: dict[str, str],
    frontier_runbook_card: dict[str, str],
) -> dict[str, str]:
    if not tokenization_handoff or not frontier_runbook_card:
        return {}

    handoff_status = str(
        tokenization_handoff.get("handoff_status", "tokenization_gain_frontier_fill_handoff_pending")
    )
    queue_status = str(tokenization_handoff.get("queue_status", "queue_in_progress"))
    ready = handoff_status == "tokenization_gain_frontier_fill_handoff_ready" and queue_status == "queue_complete"
    adapted_count = str(tokenization_handoff.get("adapted_and_aligned_count", "0"))
    case_count = str(tokenization_handoff.get("case_count", "0"))
    handoff_note = str(tokenization_handoff.get("handoff_note", ""))

    return {
        "runbook_status": (
            "tokenization_gain_frontier_fill_runbook_ready"
            if ready
            else "tokenization_gain_frontier_fill_runbook_pending"
        ),
        "recommended_frontier": str(frontier_runbook_card.get("recommended_frontier", "meeteval_compatibility")),
        "adapted_case_ratio": f"{adapted_count}/{case_count}",
        "handoff_goal": str(tokenization_handoff.get("handoff_goal", "")),
        "next_action": str(frontier_runbook_card.get("recommended_action", "")),
        "required_evidence": str(frontier_runbook_card.get("required_evidence", "")),
        "completion_signal": str(frontier_runbook_card.get("completion_signal", "")),
        "guardrail_note": (
            f"{handoff_note} Full MeetEval benchmark completion is not claimed by this runbook card."
        ).strip(),
    }


def build_runbook_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# MeetEval Tokenization Gain Frontier Fill Runbook Card",
        "",
        "This generated runbook card turns the tokenization gain handoff into a concrete frontier fill action. "
        "It remains experimental/frontier coordination only and does not claim full MeetEval benchmark completion.",
        "",
        f"- Runbook status: `{row['runbook_status']}`",
        f"- Recommended frontier: `{row['recommended_frontier']}`",
        f"- Adapted case ratio: `{row['adapted_case_ratio']}`",
        f"- Handoff goal: {row['handoff_goal']}",
        f"- Next action: `{row['next_action']}`",
        f"- Required evidence: `{row['required_evidence']}`",
        f"- Completion signal: `{row['completion_signal']}`",
        f"- Guardrail note: {row['guardrail_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_tokenization_gain_frontier_fill_runbook_card.csv"
    json_path = tables_dir / "meeteval_tokenization_gain_frontier_fill_runbook_card.json"
    md_path = figures_dir / "meeteval_tokenization_gain_frontier_fill_runbook_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RUNBOOK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_runbook_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_runbook_card_row(load_tokenization_handoff(), load_frontier_runbook_card())
    if not row:
        print("Tokenization handoff or frontier runbook card not found; tokenization runbook card not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote MeetEval tokenization gain frontier fill runbook card CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval tokenization gain frontier fill runbook card JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval tokenization gain frontier fill runbook card note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Runbook status: {row['runbook_status']}")


if __name__ == "__main__":
    main()
