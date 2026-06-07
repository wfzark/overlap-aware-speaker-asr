from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "complete_count",
    "total_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_official_execution_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_official_execution.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(execution_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not execution_rows:
        return []
    total_count = len(execution_rows)
    complete_count = sum(
        1
        for row in execution_rows
        if row.get("execution_status") == "official_cpwer_narrow_dry_run_complete"
    )
    first_case = str(execution_rows[0].get("case_id", "NoOverlap"))
    return [
        {
            "checklist_order": "1",
            "complete_count": str(complete_count),
            "total_count": str(total_count),
            "prerequisite_artifact": "results/figures/meeteval_cpwer_official_execution.md",
            "receipt_target": "results/tables/meeteval_cpwer_execution_receipt.json",
            "checklist_goal": (
                f"Verify official MeetEval cpWER narrow dry-run output before receipt writeback for {first_case}."
            ),
            "bridge_note": (
                f"Official execution reports {complete_count}/{total_count} cases with "
                "official_cpwer_narrow_dry_run_complete; confirm alignment before receipt fill."
            ),
            "next_gate": "Confirm this bridge before updating meeteval_cpwer_execution_receipt.json.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Official Execution Bridge Checklist",
        "",
        "This generated checklist connects official MeetEval cpWER narrow dry-run output to the execution receipt. "
        "It does not claim full MeetEval benchmark completion.",
        "",
        "| checklist_order | complete_count | total_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['complete_count']} | {row['total_count']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_official_execution_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_cpwer_official_execution_bridge_checklist.json"
    md_path = figures_dir / "meeteval_cpwer_official_execution_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_official_execution_rows())
    if not rows:
        print("Official execution output not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER official execution bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
