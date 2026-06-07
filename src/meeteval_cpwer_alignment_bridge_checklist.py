from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "scope",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_alignment_summary() -> dict[str, Any]:
    summary_path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    scope = str(summary.get("scope", "all_gold_cases"))
    matched_count = int(summary.get("matched_count", 0) or 0)
    case_count = int(summary.get("case_count", 0) or 0)
    return [
        {
            "checklist_order": "1",
            "scope": scope,
            "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment.md",
            "receipt_target": "results/figures/meeteval_cpwer_bridge_handoff.md",
            "checklist_goal": (
                f"Verify the cpWER alignment bridge for {scope} before advancing the bridge handoff."
            ),
            "bridge_note": (
                f"Alignment reports matched={matched_count}/{case_count}; confirm drift cases before opening the handoff target."
            ),
            "next_gate": "Confirm this bridge before opening the cpWER bridge handoff target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Bridge Checklist",
        "",
        "This generated checklist turns the alignment audit into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim a finished MeetEval evaluation.",
        "",
        "| checklist_order | scope | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['scope']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_alignment_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_cpwer_alignment_bridge_checklist.json"
    md_path = figures_dir / "meeteval_cpwer_alignment_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_alignment_summary()
    rows = build_bridge_checklist_rows(summary)
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote MeetEval cpWER alignment bridge checklist CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER alignment bridge checklist JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER alignment bridge checklist note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
