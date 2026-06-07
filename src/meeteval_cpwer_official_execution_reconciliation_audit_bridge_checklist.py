from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "aligned_count",
    "total_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_reconciliation_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_official_execution_reconciliation_audit.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(reconciliation_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not reconciliation_rows:
        return []
    aligned_count = sum(1 for row in reconciliation_rows if row.get("reconciliation_status") == "aligned")
    minor_count = sum(
        1 for row in reconciliation_rows if row.get("reconciliation_status") == "minor_drift"
    )
    total_count = len(reconciliation_rows)
    return [
        {
            "checklist_order": "1",
            "aligned_count": str(aligned_count),
            "total_count": str(total_count),
            "prerequisite_artifact": "results/figures/meeteval_cpwer_official_execution_reconciliation_audit.md",
            "receipt_target": "results/figures/meeteval_cpwer_character_level_official_execution.md",
            "checklist_goal": (
                "Verify reconciliation audit before treating character-spaced cpWER as bridge-lite comparable."
            ),
            "bridge_note": (
                f"Reconciliation audit reports {aligned_count}/{total_count} aligned and "
                f"{minor_count}/{total_count} minor drift after character tokenization."
            ),
            "next_gate": "Confirm this bridge before promoting character-spaced cpWER as the preferred frontier metric.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Official Execution Reconciliation Audit Bridge Checklist",
        "",
        "This generated checklist connects the reconciliation audit to character-level official execution evidence. "
        "It does not claim full MeetEval benchmark completion.",
        "",
        "| checklist_order | aligned_count | total_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['aligned_count']} | {row['total_count']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_official_execution_reconciliation_audit_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_cpwer_official_execution_reconciliation_audit_bridge_checklist.json"
    md_path = figures_dir / "meeteval_cpwer_official_execution_reconciliation_audit_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_reconciliation_rows())
    if not rows:
        print("Reconciliation audit not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER official execution reconciliation audit bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution reconciliation audit bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution reconciliation audit bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
