from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


HANDOFF_COLUMNS = [
    "handoff_status",
    "case_id",
    "mismatched_speaker_count",
    "dominant_blocker",
    "redistribution_diagnostic_target",
    "handoff_goal",
    "expected_evidence",
    "handoff_note",
]


def load_granularity_summary() -> dict[str, Any]:
    summary_path = (
        PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_summary.json"
    )
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_handoff_row(summary: dict[str, Any]) -> dict[str, str]:
    case_id = str(summary.get("case_id", "HeavyOverlap"))
    mismatched_speaker_count = str(summary.get("mismatched_speaker_count", "0"))
    dominant_blocker = str(summary.get("dominant_blocker", "unknown"))
    return {
        "handoff_status": "granularity_handoff_ready",
        "case_id": case_id,
        "mismatched_speaker_count": mismatched_speaker_count,
        "dominant_blocker": dominant_blocker,
        "redistribution_diagnostic_target": (
            "results/figures/meeteval_cpwer_alignment_drift_segment_redistribution_diagnostic_summary.md"
        ),
        "handoff_goal": (
            f"Run a narrow per-speaker redistribution diagnostic for {case_id} after the granularity drift finding."
        ),
        "expected_evidence": (
            "results/tables/meeteval_cpwer_alignment_drift_segment_redistribution_diagnostic_receipt.json"
        ),
        "handoff_note": (
            "experimental/frontier granularity handoff only; reconciled alignment and cpWER execution remain pending."
        ),
    }


def build_handoff_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Granularity Diagnostic Handoff",
        "",
        "This generated handoff turns the per-speaker granularity diagnostic into the next narrow MeetEval frontier step. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| handoff_status | case_id | mismatched_speaker_count | dominant_blocker | redistribution_diagnostic_target | handoff_goal | expected_evidence | handoff_note |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        (
            f"| {row['handoff_status']} | {row['case_id']} | {row['mismatched_speaker_count']} | "
            f"{row['dominant_blocker']} | {row['redistribution_diagnostic_target']} | {row['handoff_goal']} | "
            f"{row['expected_evidence']} | {row['handoff_note']} |"
        ),
    ]
    return lines


def build_handoff_receipt_rows(handoff_row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "execution_status": "handoff_documented",
            "handoff_scope": "single_drift_case_granularity_to_redistribution",
            "case_id": str(handoff_row.get("case_id", "")),
            "writeback_note": (
                "Granularity diagnostic handoff documented for coordination; "
                "per-speaker redistribution diagnostic and cpWER execution remain pending."
            ),
        }
    ]


def build_handoff_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Granularity Diagnostic Handoff Receipt",
        "",
        "This receipt records the granularity diagnostic handoff writeback. It does not claim cpWER execution.",
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

    csv_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_handoff.csv"
    json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_handoff.json"
    md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_handoff.md"
    receipt_json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_handoff_receipt.json"
    receipt_md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_handoff_receipt.md"

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
    summary = load_granularity_summary()
    handoff_row = build_handoff_row(summary)
    receipt_rows = build_handoff_receipt_rows(handoff_row)
    csv_path, json_path, md_path, receipt_json_path, receipt_md_path = write_outputs(
        handoff_row,
        receipt_rows,
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic handoff CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic handoff JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic handoff note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic handoff receipt JSON: "
        f"{receipt_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic handoff receipt note: "
        f"{receipt_md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
