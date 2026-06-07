from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .export_meeteval_compatibility import load_hypothesis_payload
from .meeteval_dry_run import load_jsonl_segments, select_preferred_case, time_ranges_valid


PREFLIGHT_COLUMNS = [
    "case_id",
    "handoff_status",
    "scaffold_status",
    "hypothesis_source",
    "reference_segment_count",
    "hypothesis_segment_count",
    "speaker_set_match",
    "time_range_valid",
    "export_path_valid",
    "preflight_pass",
    "preflight_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "run_scope",
    "case_id",
    "hypothesis_source",
    "preflight_pass",
    "expected_inputs",
    "expected_outputs",
    "writeback_note",
]


def load_execution_handoff() -> dict[str, Any]:
    handoff_path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_execution_handoff.json"
    if not handoff_path.exists():
        return {}
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def extract_speakers(segments: list[dict[str, Any]]) -> set[str]:
    return {
        str(segment.get("speaker", "")).strip()
        for segment in segments
        if str(segment.get("speaker", "")).strip()
    }


def run_preflight(case_id: str, handoff: dict[str, Any]) -> dict[str, Any]:
    reference_path = PROJECT_ROOT / "results" / "tables" / "meeteval_reference_segments.jsonl"
    hypothesis_path = PROJECT_ROOT / "results" / "tables" / "meeteval_hypothesis_segments.jsonl"
    reference_segments = load_jsonl_segments(reference_path, case_id)
    hypothesis_segments = load_jsonl_segments(hypothesis_path, case_id)
    export_path_valid = bool(reference_segments and hypothesis_segments)
    speaker_set_match = extract_speakers(reference_segments) == extract_speakers(hypothesis_segments)
    time_range_valid = time_ranges_valid(reference_segments) and time_ranges_valid(hypothesis_segments)
    preflight_pass = export_path_valid and speaker_set_match and time_range_valid

    hypothesis_source = ""
    try:
        hypothesis_source = str(load_hypothesis_payload(case_id).get("hypothesis_source", ""))
    except FileNotFoundError:
        hypothesis_source = "unknown"

    if preflight_pass:
        preflight_note = (
            f"Execution preflight validated for {case_id}; segment exports align on speakers and time ranges. "
            "Official cpWER has not been computed."
        )
    else:
        preflight_note = (
            f"Execution preflight for {case_id} found issues; review segment exports before official cpWER execution."
        )

    return {
        "case_id": case_id,
        "handoff_status": str(handoff.get("handoff_status", "execution_handoff_ready")),
        "scaffold_status": str(handoff.get("scaffold_status", "scaffold_only")),
        "hypothesis_source": hypothesis_source,
        "reference_segment_count": len(reference_segments),
        "hypothesis_segment_count": len(hypothesis_segments),
        "speaker_set_match": speaker_set_match,
        "time_range_valid": time_range_valid,
        "export_path_valid": export_path_valid,
        "preflight_pass": preflight_pass,
        "preflight_note": preflight_note,
    }


def build_preflight_lines(row: dict[str, Any]) -> list[str]:
    lines = [
        "# MeetEval cpWER Execution Preflight",
        "",
        "This generated note records a narrow execution preflight for one verified gold case. "
        "It does not claim official cpWER evaluation or benchmark completion.",
        "",
        "| case_id | handoff_status | scaffold_status | hypothesis_source | reference_segment_count | "
        "hypothesis_segment_count | speaker_set_match | time_range_valid | export_path_valid | preflight_pass | preflight_note |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
        (
            f"| {row['case_id']} | {row['handoff_status']} | {row['scaffold_status']} | {row['hypothesis_source']} | "
            f"{row['reference_segment_count']} | {row['hypothesis_segment_count']} | {row['speaker_set_match']} | "
            f"{row['time_range_valid']} | {row['export_path_valid']} | {row['preflight_pass']} | {row['preflight_note']} |"
        ),
    ]
    return lines


def build_receipt_rows(preflight_row: dict[str, Any]) -> list[dict[str, str]]:
    status = "preflight_complete" if preflight_row.get("preflight_pass") else "preflight_failed"
    return [
        {
            "execution_status": status,
            "run_scope": "single_case_cpwer_execution_preflight",
            "case_id": str(preflight_row.get("case_id", "")),
            "hypothesis_source": str(preflight_row.get("hypothesis_source", "")),
            "preflight_pass": str(preflight_row.get("preflight_pass", False)),
            "expected_inputs": (
                "results/tables/meeteval_reference_segments.jsonl; "
                "results/tables/meeteval_hypothesis_segments.jsonl"
            ),
            "expected_outputs": "Official cpWER score receipt for one verified gold case.",
            "writeback_note": (
                "Execution preflight documented; official MeetEval cpWER evaluation remains pending."
            ),
        }
    ]


def build_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Execution Preflight Receipt",
        "",
        "This receipt records the execution preflight writeback. It does not claim cpWER execution.",
        "",
        "| execution_status | run_scope | case_id | hypothesis_source | preflight_pass | expected_inputs | expected_outputs | writeback_note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['run_scope']} | {row['case_id']} | {row['hypothesis_source']} | "
            f"{row['preflight_pass']} | {row['expected_inputs']} | {row['expected_outputs']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    preflight_row: dict[str, Any],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_execution_preflight.csv"
    json_path = tables_dir / "meeteval_cpwer_execution_preflight.json"
    md_path = figures_dir / "meeteval_cpwer_execution_preflight.md"
    receipt_json_path = tables_dir / "meeteval_cpwer_execution_preflight_receipt.json"
    receipt_md_path = figures_dir / "meeteval_cpwer_execution_preflight_receipt.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PREFLIGHT_COLUMNS)
        writer.writeheader()
        writer.writerow(preflight_row)
    json_path.write_text(json.dumps(preflight_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_preflight_lines(preflight_row)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, receipt_json_path, receipt_md_path


def resolve_execution_case_id(handoff: dict[str, Any]) -> str:
    case_id = str(handoff.get("case_id", "NoOverlap"))
    if case_id in {"", "ALL"}:
        checklist_path = PROJECT_ROOT / "results" / "tables" / "meeteval_dry_run_checklist.csv"
        return select_preferred_case(checklist_path)
    return case_id


def main() -> None:
    handoff = load_execution_handoff()
    case_id = resolve_execution_case_id(handoff)
    preflight_row = run_preflight(case_id, handoff)
    receipt_rows = build_receipt_rows(preflight_row)
    csv_path, json_path, md_path, receipt_json_path, receipt_md_path = write_outputs(
        preflight_row,
        receipt_rows,
    )
    print(f"Wrote MeetEval cpWER execution preflight CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER execution preflight JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER execution preflight note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER execution preflight receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER execution preflight receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
