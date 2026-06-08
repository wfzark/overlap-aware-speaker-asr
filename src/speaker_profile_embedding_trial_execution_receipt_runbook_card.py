from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


RUNBOOK_COLUMNS = [
    "recommended_case",
    "readiness_status",
    "receipt_target",
    "recommended_action",
    "required_evidence",
    "completion_signal",
    "runbook_note",
]


def load_operator_brief() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_receipt_operator_brief.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_bridge_checklist_rows() -> list[dict[str, str]]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge_checklist.json"
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_runbook_card_row(
    operator_brief: dict[str, str],
    bridge_checklist_row: dict[str, str],
) -> dict[str, str]:
    if not operator_brief:
        return {}
    recommended_case = str(operator_brief.get("operator_case", "NoOverlap"))
    readiness_status = str(operator_brief.get("operator_status", "receipt_not_ready"))
    receipt_target = str(operator_brief.get("operator_target", ""))
    checklist_order = str(bridge_checklist_row.get("checklist_order", "1"))
    return {
        "recommended_case": recommended_case,
        "readiness_status": readiness_status,
        "receipt_target": receipt_target,
        "recommended_action": str(operator_brief.get("operator_action", "")),
        "required_evidence": str(operator_brief.get("operator_evidence", "")),
        "completion_signal": str(
            bridge_checklist_row.get(
                "next_gate",
                f"Confirm this bridge before opening {receipt_target}.",
            )
        ),
        "runbook_note": (
            f"Start with {recommended_case} because readiness_status={readiness_status} and "
            f"checklist_order={checklist_order}. This remains experimental/frontier coordination only "
            "and does not fill the receipt or claim voiceprint success."
        ),
    }


def build_runbook_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Embedding Trial Execution Receipt Runbook Card",
        "",
        "This generated runbook card condenses the current speaker-profile receipt action into a one-page execution card. "
        "It remains experimental/frontier coordination only and does not fill receipts or claim voiceprint success.",
        "",
        f"- Recommended case: `{row['recommended_case']}`",
        f"- Readiness status: `{row['readiness_status']}`",
        f"- Receipt target: `{row['receipt_target']}`",
        f"- Recommended action: `{row['recommended_action']}`",
        f"- Required evidence: `{row['required_evidence']}`",
        f"- Completion signal: `{row['completion_signal']}`",
        f"- Runbook note: {row['runbook_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_runbook_card.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_runbook_card.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_runbook_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RUNBOOK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_runbook_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    bridge_rows = load_bridge_checklist_rows()
    row = build_runbook_card_row(load_operator_brief(), bridge_rows[0] if bridge_rows else {})
    if not row:
        print("Speaker-profile receipt operator brief not found; runbook card not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(
        "Wrote speaker profile embedding trial execution receipt runbook card CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt runbook card JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt runbook card note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
