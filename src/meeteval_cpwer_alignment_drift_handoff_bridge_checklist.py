from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "case_id",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_drift_handoff_row() -> dict[str, str]:
    handoff_path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_drift_handoff.json"
    if not handoff_path.exists():
        return {}
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[0]
    return {}


def build_bridge_checklist_rows(handoff_row: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(handoff_row.get("case_id", "HeavyOverlap"))
    drift_severity = str(handoff_row.get("drift_severity", "moderate"))
    return [
        {
            "checklist_order": "1",
            "case_id": case_id,
            "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment_drift_handoff.md",
            "receipt_target": "results/figures/meeteval_cpwer_alignment_drift_bridge_checklist.md",
            "checklist_goal": (
                f"Verify the drift handoff bridge for {case_id} before reopening the alignment bridge checklist."
            ),
            "bridge_note": (
                f"Drift handoff remains severity={drift_severity}; confirm segment-inspection context before advancing."
            ),
            "next_gate": "Confirm this bridge before opening the cpWER alignment drift bridge checklist target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Handoff Bridge Checklist",
        "",
        "This generated checklist turns the drift handoff into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim cpWER execution.",
        "",
        "| checklist_order | case_id | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['case_id']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_alignment_drift_handoff_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_cpwer_alignment_drift_handoff_bridge_checklist.json"
    md_path = figures_dir / "meeteval_cpwer_alignment_drift_handoff_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    handoff_row = load_drift_handoff_row()
    rows = build_bridge_checklist_rows(handoff_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER alignment drift handoff bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift handoff bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift handoff bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
