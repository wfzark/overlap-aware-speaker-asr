from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


HANDOFF_COLUMNS = [
    "handoff_status",
    "case_id",
    "scaffold_status",
    "reconciliation_target",
    "handoff_goal",
    "expected_evidence",
    "handoff_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "handoff_scope",
    "case_id",
    "writeback_note",
]


def load_reconciliation_scaffold() -> dict[str, Any]:
    scaffold_path = (
        PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_drift_segment_reconciliation_scaffold.json"
    )
    if not scaffold_path.exists():
        return {}
    payload = json.loads(scaffold_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_handoff_row(scaffold: dict[str, Any]) -> dict[str, str]:
    case_id = str(scaffold.get("case_id", "HeavyOverlap"))
    scaffold_status = str(scaffold.get("scaffold_status", "scaffold_only"))
    reconciliation_target = str(scaffold.get("reconciliation_target", ""))
    return {
        "handoff_status": "reconciliation_handoff_ready",
        "case_id": case_id,
        "scaffold_status": scaffold_status,
        "reconciliation_target": reconciliation_target,
        "handoff_goal": (
            f"Run a narrow segment reconciliation diagnostic for {case_id} before any cpWER bridge advance."
        ),
        "expected_evidence": str(scaffold.get("expected_evidence", "")),
        "handoff_note": (
            "experimental/frontier reconciliation handoff only; reconciled alignment and cpWER execution remain pending."
        ),
    }


def build_handoff_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Reconciliation Handoff",
        "",
        "This generated handoff turns the reconciliation scaffold into the next narrow MeetEval frontier step. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| handoff_status | case_id | scaffold_status | reconciliation_target | handoff_goal | expected_evidence | handoff_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['handoff_status']} | {row['case_id']} | {row['scaffold_status']} | {row['reconciliation_target']} | "
            f"{row['handoff_goal']} | {row['expected_evidence']} | {row['handoff_note']} |"
        ),
    ]
    return lines


def build_handoff_receipt_rows(handoff_row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "execution_status": "handoff_documented",
            "handoff_scope": "single_drift_case_reconciliation",
            "case_id": str(handoff_row.get("case_id", "")),
            "writeback_note": (
                "Reconciliation handoff documented for coordination; reconciled alignment and cpWER execution remain pending."
            ),
        }
    ]


def build_handoff_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Reconciliation Handoff Receipt",
        "",
        "This receipt records the reconciliation handoff writeback. It does not claim cpWER execution.",
        "",
        "| execution_status | handoff_scope | case_id | writeback_note |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['handoff_scope']} | {row['case_id']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    handoff_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_handoff.csv"
    json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_handoff.json"
    md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_handoff.md"
    receipt_json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_handoff_receipt.json"
    receipt_md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_handoff_receipt.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerow(handoff_row)
    json_path.write_text(json.dumps(handoff_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_handoff_lines(handoff_row)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_handoff_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, receipt_json_path, receipt_md_path


def main() -> None:
    scaffold = load_reconciliation_scaffold()
    handoff_row = build_handoff_row(scaffold)
    receipt_rows = build_handoff_receipt_rows(handoff_row)
    csv_path, json_path, md_path, receipt_json_path, receipt_md_path = write_outputs(
        handoff_row,
        receipt_rows,
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation handoff CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation handoff JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation handoff note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation handoff receipt JSON: "
        f"{receipt_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation handoff receipt note: "
        f"{receipt_md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
