from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .export_meeteval_compatibility import load_hypothesis_payload
from .meeteval_dry_run import (
    extract_speakers,
    load_jsonl_segments,
    time_ranges_valid,
)


DIAGNOSTIC_COLUMNS = [
    "case_id",
    "hypothesis_source",
    "reference_segment_count",
    "hypothesis_segment_count",
    "speaker_segment_count_match",
    "speaker_set_match",
    "time_range_valid",
    "export_path_valid",
    "reconciliation_pass",
    "diagnostic_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "diagnostic_scope",
    "case_id",
    "reconciliation_pass",
    "writeback_note",
]


def load_reconciliation_handoff_row() -> dict[str, Any]:
    handoff_path = (
        PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_drift_segment_reconciliation_handoff.json"
    )
    if not handoff_path.exists():
        return {}
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def count_segments_per_speaker(segments: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for segment in segments:
        speaker = str(segment.get("speaker", "")).strip()
        if speaker:
            counts[speaker] += 1
    return counts


def speaker_segment_counts_match(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
) -> bool:
    return count_segments_per_speaker(reference_segments) == count_segments_per_speaker(hypothesis_segments)


def run_reconciliation_diagnostic(case_id: str) -> dict[str, Any]:
    reference_path = PROJECT_ROOT / "results" / "tables" / "meeteval_reference_segments.jsonl"
    hypothesis_path = PROJECT_ROOT / "results" / "tables" / "meeteval_hypothesis_segments.jsonl"
    reference_segments = load_jsonl_segments(reference_path, case_id)
    hypothesis_segments = load_jsonl_segments(hypothesis_path, case_id)
    export_path_valid = bool(reference_segments and hypothesis_segments)
    speaker_set_match = extract_speakers(reference_segments) == extract_speakers(hypothesis_segments)
    speaker_segment_count_match = speaker_segment_counts_match(reference_segments, hypothesis_segments)
    time_range_valid = time_ranges_valid(reference_segments) and time_ranges_valid(hypothesis_segments)
    reconciliation_pass = (
        export_path_valid
        and speaker_set_match
        and speaker_segment_count_match
        and time_range_valid
    )

    hypothesis_source = ""
    try:
        hypothesis_source = str(load_hypothesis_payload(case_id).get("hypothesis_source", ""))
    except FileNotFoundError:
        hypothesis_source = "unknown"

    if reconciliation_pass:
        diagnostic_note = (
            f"Reconciliation readiness validated for drift case {case_id}; per-speaker segment counts align. "
            "Reconciled alignment and cpWER execution remain pending."
        )
    else:
        diagnostic_note = (
            f"Reconciliation readiness check for drift case {case_id} found issues; "
            "review per-speaker segment structure before any cpWER claim."
        )

    return {
        "case_id": case_id,
        "hypothesis_source": hypothesis_source,
        "reference_segment_count": len(reference_segments),
        "hypothesis_segment_count": len(hypothesis_segments),
        "speaker_segment_count_match": speaker_segment_count_match,
        "speaker_set_match": speaker_set_match,
        "time_range_valid": time_range_valid,
        "export_path_valid": export_path_valid,
        "reconciliation_pass": reconciliation_pass,
        "diagnostic_note": diagnostic_note,
    }


def build_diagnostic_receipt_row(diagnostic: dict[str, Any]) -> dict[str, str]:
    return {
        "execution_status": "reconciliation_diagnostic_complete",
        "diagnostic_scope": "single_drift_case",
        "case_id": str(diagnostic.get("case_id", "")),
        "reconciliation_pass": str(diagnostic.get("reconciliation_pass", False)),
        "writeback_note": (
            "Narrow reconciliation diagnostic complete for the drift case. "
            "Reconciled alignment and cpWER execution remain pending."
        ),
    }


def build_diagnostic_summary_lines(diagnostic: dict[str, Any]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Reconciliation Diagnostic",
        "",
        "This generated note records the first narrow reconciliation diagnostic for the drift handoff case. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| case_id | hypothesis_source | reference_segment_count | hypothesis_segment_count | speaker_segment_count_match | speaker_set_match | time_range_valid | export_path_valid | reconciliation_pass | diagnostic_note |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |",
        (
            f"| {diagnostic['case_id']} | {diagnostic['hypothesis_source']} | "
            f"{diagnostic['reference_segment_count']} | {diagnostic['hypothesis_segment_count']} | "
            f"{diagnostic['speaker_segment_count_match']} | {diagnostic['speaker_set_match']} | "
            f"{diagnostic['time_range_valid']} | {diagnostic['export_path_valid']} | "
            f"{diagnostic['reconciliation_pass']} | {diagnostic['diagnostic_note']} |"
        ),
    ]
    return lines


def build_diagnostic_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Reconciliation Diagnostic Receipt",
        "",
        "This receipt records the reconciliation diagnostic writeback. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| execution_status | diagnostic_scope | case_id | reconciliation_pass | writeback_note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['diagnostic_scope']} | {row['case_id']} | "
            f"{row['reconciliation_pass']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    diagnostic: dict[str, Any],
    receipt_row: dict[str, str],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic.csv"
    json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic.json"
    md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic.md"
    receipt_json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic_receipt.json"
    receipt_md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic_receipt.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        writer.writerow(diagnostic)
    json_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_diagnostic_summary_lines(diagnostic)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps([receipt_row], ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_diagnostic_receipt_lines([receipt_row])) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, receipt_json_path, receipt_md_path


def main() -> None:
    handoff_row = load_reconciliation_handoff_row()
    case_id = str(handoff_row.get("case_id", "HeavyOverlap"))
    diagnostic = run_reconciliation_diagnostic(case_id)
    receipt_row = build_diagnostic_receipt_row(diagnostic)
    csv_path, json_path, md_path, receipt_json_path, receipt_md_path = write_outputs(
        diagnostic,
        receipt_row,
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation diagnostic CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation diagnostic JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation diagnostic note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation diagnostic receipt JSON: "
        f"{receipt_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment reconciliation diagnostic receipt note: "
        f"{receipt_md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Reconciliation pass: {diagnostic['reconciliation_pass']}")


if __name__ == "__main__":
    main()
