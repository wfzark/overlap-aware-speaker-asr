from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .export_meeteval_compatibility import load_hypothesis_payload


DIAGNOSTIC_COLUMNS = [
    "case_id",
    "hypothesis_source",
    "reference_segment_count",
    "hypothesis_segment_count",
    "speaker_set_match",
    "time_range_valid",
    "export_path_valid",
    "diagnostic_pass",
    "diagnostic_note",
]

RECEIPT_DIAGNOSTIC_COLUMNS = [
    "execution_status",
    "run_scope",
    "case_id",
    "hypothesis_source",
    "expected_inputs",
    "reference_segment_count",
    "hypothesis_segment_count",
    "speaker_set_match",
    "time_range_valid",
    "export_path_valid",
    "diagnostic_pass",
    "expected_outputs",
    "writeback_note",
]


def load_jsonl_segments(path: Path, case_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    segments: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if str(payload.get("session_id", "")) == case_id:
            segments.append(payload)
    return segments


def extract_speakers(segments: list[dict[str, Any]]) -> set[str]:
    return {
        str(segment.get("speaker", "")).strip()
        for segment in segments
        if str(segment.get("speaker", "")).strip()
    }


def time_ranges_valid(segments: list[dict[str, Any]]) -> bool:
    if not segments:
        return False
    for segment in segments:
        start = float(segment.get("start_time", 0.0) or 0.0)
        end = float(segment.get("end_time", 0.0) or 0.0)
        if end < start:
            return False
    return True


def select_preferred_case(checklist_path: Path) -> str:
    if checklist_path.exists():
        with checklist_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("dry_run_priority", "")) == "preferred":
                    return str(row.get("case_id", ""))
    return "NoOverlap"


def run_diagnostic(case_id: str) -> dict[str, Any]:
    reference_path = PROJECT_ROOT / "results" / "tables" / "meeteval_reference_segments.jsonl"
    hypothesis_path = PROJECT_ROOT / "results" / "tables" / "meeteval_hypothesis_segments.jsonl"
    reference_segments = load_jsonl_segments(reference_path, case_id)
    hypothesis_segments = load_jsonl_segments(hypothesis_path, case_id)
    export_path_valid = bool(reference_segments and hypothesis_segments)
    speaker_set_match = extract_speakers(reference_segments) == extract_speakers(hypothesis_segments)
    time_range_valid = time_ranges_valid(reference_segments) and time_ranges_valid(hypothesis_segments)
    diagnostic_pass = export_path_valid and speaker_set_match and time_range_valid

    hypothesis_source = ""
    try:
        hypothesis_source = str(load_hypothesis_payload(case_id).get("hypothesis_source", ""))
    except FileNotFoundError:
        hypothesis_source = "unknown"

    if diagnostic_pass:
        diagnostic_note = (
            f"Export path validated for {case_id}; reference and hypothesis segments align on speakers and time ranges. "
            "cpWER has not been computed."
        )
    else:
        diagnostic_note = (
            f"Export path check for {case_id} found issues; review segment exports before any cpWER claim."
        )

    return {
        "case_id": case_id,
        "hypothesis_source": hypothesis_source,
        "reference_segment_count": len(reference_segments),
        "hypothesis_segment_count": len(hypothesis_segments),
        "speaker_set_match": speaker_set_match,
        "time_range_valid": time_range_valid,
        "export_path_valid": export_path_valid,
        "diagnostic_pass": diagnostic_pass,
        "diagnostic_note": diagnostic_note,
    }


def build_diagnostic_receipt_row(diagnostic: dict[str, Any]) -> dict[str, str]:
    return {
        "execution_status": "diagnostic_complete",
        "run_scope": "single_verified_case",
        "case_id": str(diagnostic.get("case_id", "")),
        "hypothesis_source": str(diagnostic.get("hypothesis_source", "")),
        "expected_inputs": "results/tables/meeteval_reference_segments.jsonl; results/tables/meeteval_hypothesis_segments.jsonl",
        "reference_segment_count": str(diagnostic.get("reference_segment_count", 0)),
        "hypothesis_segment_count": str(diagnostic.get("hypothesis_segment_count", 0)),
        "speaker_set_match": str(diagnostic.get("speaker_set_match", False)),
        "time_range_valid": str(diagnostic.get("time_range_valid", False)),
        "export_path_valid": str(diagnostic.get("export_path_valid", False)),
        "diagnostic_pass": str(diagnostic.get("diagnostic_pass", False)),
        "expected_outputs": str(diagnostic.get("diagnostic_note", "")),
        "writeback_note": "Diagnostic dry run complete. MeetEval / cpWER evaluation still pending.",
    }


def build_diagnostic_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval Dry Run Receipt",
        "",
        "This receipt records the first narrow diagnostic dry run. It does not claim a finished MeetEval or cpWER evaluation.",
        "",
        "| execution_status | run_scope | case_id | hypothesis_source | reference_segment_count | hypothesis_segment_count | speaker_set_match | time_range_valid | export_path_valid | diagnostic_pass | writeback_note |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['run_scope']} | {row['case_id']} | {row['hypothesis_source']} | "
            f"{row['reference_segment_count']} | {row['hypothesis_segment_count']} | {row['speaker_set_match']} | "
            f"{row['time_range_valid']} | {row['export_path_valid']} | {row['diagnostic_pass']} | {row['writeback_note']} |"
        )
    return lines


def build_diagnostic_summary_lines(diagnostic: dict[str, Any]) -> list[str]:
    lines = [
        "# MeetEval Dry Run Diagnostic",
        "",
        "This generated note records the first narrow diagnostic dry run on the preferred verified case. "
        "It does not claim a finished MeetEval or cpWER evaluation.",
        "",
        "| case_id | hypothesis_source | reference_segment_count | hypothesis_segment_count | speaker_set_match | time_range_valid | export_path_valid | diagnostic_pass | diagnostic_note |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
        (
            f"| {diagnostic['case_id']} | {diagnostic['hypothesis_source']} | {diagnostic['reference_segment_count']} | "
            f"{diagnostic['hypothesis_segment_count']} | {diagnostic['speaker_set_match']} | {diagnostic['time_range_valid']} | "
            f"{diagnostic['export_path_valid']} | {diagnostic['diagnostic_pass']} | {diagnostic['diagnostic_note']} |"
        ),
    ]
    return lines


def write_outputs(
    diagnostic: dict[str, Any],
    receipt_row: dict[str, str],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_csv_path = tables_dir / "meeteval_dry_run_diagnostic.csv"
    diagnostic_json_path = tables_dir / "meeteval_dry_run_diagnostic.json"
    diagnostic_md_path = figures_dir / "meeteval_dry_run_diagnostic.md"
    receipt_json_path = tables_dir / "meeteval_dry_run_receipt.json"
    receipt_md_path = figures_dir / "meeteval_dry_run_receipt.md"

    with diagnostic_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        writer.writerow(diagnostic)
    diagnostic_json_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostic_md_path.write_text("\n".join(build_diagnostic_summary_lines(diagnostic)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps([receipt_row], ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_diagnostic_receipt_lines([receipt_row])) + "\n", encoding="utf-8")
    return diagnostic_csv_path, diagnostic_json_path, diagnostic_md_path, receipt_json_path, receipt_md_path


def main() -> None:
    checklist_path = PROJECT_ROOT / "results" / "tables" / "meeteval_dry_run_checklist.csv"
    case_id = select_preferred_case(checklist_path)
    diagnostic = run_diagnostic(case_id)
    receipt_row = build_diagnostic_receipt_row(diagnostic)
    (
        diagnostic_csv_path,
        diagnostic_json_path,
        diagnostic_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(diagnostic, receipt_row)
    print(f"Wrote MeetEval dry run diagnostic CSV: {diagnostic_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run diagnostic JSON: {diagnostic_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run diagnostic note: {diagnostic_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Diagnostic pass: {diagnostic['diagnostic_pass']}")


if __name__ == "__main__":
    main()
