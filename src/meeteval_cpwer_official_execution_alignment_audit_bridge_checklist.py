from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "drift_case_count",
    "total_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_alignment_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_official_execution_alignment_audit.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(alignment_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not alignment_rows:
        return []
    drift_count = sum(
        1 for row in alignment_rows if row.get("alignment_status") in {"moderate_drift", "minor_drift"}
    )
    total_count = len(alignment_rows)
    return [
        {
            "checklist_order": "1",
            "drift_case_count": str(drift_count),
            "total_count": str(total_count),
            "prerequisite_artifact": "results/figures/meeteval_cpwer_official_execution_alignment_audit.md",
            "receipt_target": "results/figures/meeteval_cpwer_official_execution_tokenization_diagnostic.md",
            "checklist_goal": (
                "Verify alignment drift findings before opening the tokenization diagnostic."
            ),
            "bridge_note": (
                f"Alignment audit reports {drift_count}/{total_count} drift cases; "
                "confirm before diagnosing Chinese tokenization root cause."
            ),
            "next_gate": "Confirm this bridge before running the tokenization diagnostic.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Official Execution Alignment Audit Bridge Checklist",
        "",
        "This generated checklist connects the alignment audit to the tokenization diagnostic. "
        "It does not claim benchmark completion.",
        "",
        "| checklist_order | drift_case_count | total_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['drift_case_count']} | {row['total_count']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_official_execution_alignment_audit_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_cpwer_official_execution_alignment_audit_bridge_checklist.json"
    md_path = figures_dir / "meeteval_cpwer_official_execution_alignment_audit_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_alignment_rows())
    if not rows:
        print("Alignment audit not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER official execution alignment audit bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution alignment audit bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution alignment audit bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
