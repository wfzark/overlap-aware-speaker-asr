from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "case_id",
    "inspection_status",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_segment_inspection() -> dict[str, str]:
    inspection_path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_drift_segment_inspection.json"
    if not inspection_path.exists():
        return {}
    payload = json.loads(inspection_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(inspection: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(inspection.get("case_id", "HeavyOverlap"))
    inspection_pass = inspection.get("inspection_pass", False)
    passed = inspection_pass if isinstance(inspection_pass, bool) else str(inspection_pass).lower() == "true"
    inspection_status = "segment_inspection_complete" if passed else "segment_inspection_pending"
    return [
        {
            "checklist_order": "1",
            "case_id": case_id,
            "inspection_status": inspection_status,
            "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment_drift_segment_inspection.md",
            "receipt_target": "results/figures/meeteval_cpwer_alignment_drift_segment_handoff_bridge_checklist.md",
            "checklist_goal": (
                f"Verify the segment inspection bridge for {case_id} before reopening the segment handoff bridge checklist."
            ),
            "bridge_note": (
                f"Segment inspection status={inspection_status}; confirm drift context before advancing the handoff bridge."
            ),
            "next_gate": "Confirm this bridge before opening the cpWER segment handoff bridge checklist target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Inspection Bridge Checklist",
        "",
        "This generated checklist turns the segment inspection into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim cpWER execution.",
        "",
        "| checklist_order | case_id | inspection_status | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['case_id']} | {row['inspection_status']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_inspection_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_inspection_bridge_checklist.json"
    md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_inspection_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    inspection = load_segment_inspection()
    rows = build_bridge_checklist_rows(inspection)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER alignment drift segment inspection bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment inspection bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment inspection bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
