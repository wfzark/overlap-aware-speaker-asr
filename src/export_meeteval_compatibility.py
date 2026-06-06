from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .evaluate_cer import list_verified_cases, load_json, load_reference


SUMMARY_COLUMNS = [
    "case_id",
    "reference_segment_count",
    "hypothesis_segment_count",
    "speaker_count",
    "hypothesis_source",
    "reference_export",
    "hypothesis_export",
    "observation",
]

READINESS_COLUMNS = [
    "bridge_status",
    "case_count",
    "raw_source_count",
    "cleaned_fallback_count",
    "readiness_note",
    "next_action",
]

DRY_RUN_HANDOFF_COLUMNS = [
    "bridge_status",
    "source_mix",
    "recommended_slice",
    "dry_run_goal",
    "primary_blocker",
    "expected_evidence",
    "handoff_note",
]

DRY_RUN_RECEIPT_COLUMNS = [
    "execution_status",
    "run_scope",
    "expected_inputs",
    "expected_outputs",
    "writeback_note",
]


def load_reference_payload(case_id: str) -> dict[str, Any]:
    payload = load_reference(case_id)
    return {"segments": list(payload.get("segments", []))}


def load_hypothesis_payload(case_id: str) -> dict[str, Any]:
    raw_path = PROJECT_ROOT / "results" / "transcripts_speaker" / f"{case_id}_separated_speaker_transcript.json"
    if raw_path.exists():
        payload = load_json(raw_path)
        return {
            "segments": list(payload.get("segments", [])),
            "hypothesis_source": "separated_whisper",
        }

    cleaned_path = (
        PROJECT_ROOT / "results" / "transcripts_postprocessed" / f"{case_id}_separated_speaker_transcript_cleaned.json"
    )
    if cleaned_path.exists():
        payload = load_json(cleaned_path)
        return {
            "segments": list(payload.get("cleaned_segments", [])),
            "hypothesis_source": "separated_whisper_cleaned",
        }
    raise FileNotFoundError(
        f"Missing speaker transcript candidates: {raw_path.relative_to(PROJECT_ROOT)} and {cleaned_path.relative_to(PROJECT_ROOT)}"
    )


def build_meeteval_compatibility_rows(
    case_ids: list[str],
    reference_payloads: dict[str, dict[str, Any]],
    hypothesis_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        reference_segments = list(reference_payloads.get(case_id, {}).get("segments", []))
        hypothesis_segments = list(hypothesis_payloads.get(case_id, {}).get("segments", []))
        speakers = {
            str(segment.get("speaker", "")).strip()
            for segment in reference_segments + hypothesis_segments
            if str(segment.get("speaker", "")).strip()
        }
        rows.append(
            {
                "case_id": case_id,
                "reference_segment_count": len(reference_segments),
                "hypothesis_segment_count": len(hypothesis_segments),
                "speaker_count": len(speakers),
                "hypothesis_source": str(hypothesis_payloads.get(case_id, {}).get("hypothesis_source", "")),
                "reference_export": "results/tables/meeteval_reference_segments.jsonl",
                "hypothesis_export": "results/tables/meeteval_hypothesis_segments.jsonl",
                "observation": "compatibility bridge only; this export does not claim cpWER evaluation yet.",
            }
        )
    return rows


def build_meeteval_segment_lines(case_id: str, source: str, segments: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for segment in segments:
        payload = {
            "session_id": case_id,
            "speaker": str(segment.get("speaker", "")),
            "start_time": float(segment.get("start", 0.0) or 0.0),
            "end_time": float(segment.get("end", 0.0) or 0.0),
            "text": str(segment.get("text", "")),
            "source": source,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return lines


def build_meeteval_compatibility_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# MeetEval Compatibility Note",
        "",
        "This generated note documents a segment-level compatibility bridge only; it does not claim a finished MeetEval or cpWER evaluation.",
        "",
        "| case_id | reference_segment_count | hypothesis_segment_count | speaker_count | hypothesis_source | reference_export | hypothesis_export | observation |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['reference_segment_count']} | {row['hypothesis_segment_count']} | {row['speaker_count']} | {row['hypothesis_source']} | "
            f"{row['reference_export']} | {row['hypothesis_export']} | {row['observation']} |"
        )
    return lines


def build_meeteval_readiness_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    raw_source_count = sum(1 for row in rows if str(row.get("hypothesis_source", "")) == "separated_whisper")
    cleaned_fallback_count = sum(
        1 for row in rows if str(row.get("hypothesis_source", "")) == "separated_whisper_cleaned"
    )
    bridge_status = "ready_for_dry_run" if rows else "missing_export"
    readiness_note = (
        "The bridge is export-complete, but cleaned fallback is still common, so any next MeetEval step should stay narrow and diagnostic."
        if cleaned_fallback_count
        else "The bridge is export-complete with raw separated sources only, so a narrow diagnostic dry run is the next reasonable step."
    )
    next_action = "Use one narrow dry run before claiming any cpWER-style evaluation."
    return [
        {
            "bridge_status": bridge_status,
            "case_count": str(len(rows)),
            "raw_source_count": str(raw_source_count),
            "cleaned_fallback_count": str(cleaned_fallback_count),
            "readiness_note": readiness_note,
            "next_action": next_action,
        }
    ]


def build_meeteval_readiness_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval Readiness",
        "",
        "This generated card summarizes whether the current compatibility bridge is ready for a narrow diagnostic follow-up. It does not claim that MeetEval or cpWER has already been run.",
        "",
        "| bridge_status | case_count | raw_source_count | cleaned_fallback_count | readiness_note | next_action |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['bridge_status']} | {row['case_count']} | {row['raw_source_count']} | {row['cleaned_fallback_count']} | {row['readiness_note']} | {row['next_action']} |"
        )
    return lines


def build_meeteval_dry_run_handoff_rows(readiness_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not readiness_rows:
        return []

    readiness = readiness_rows[0]
    cleaned_fallback_count = int(str(readiness.get("cleaned_fallback_count", "0")) or "0")
    raw_source_count = int(str(readiness.get("raw_source_count", "0")) or "0")
    if cleaned_fallback_count > raw_source_count:
        source_mix = "cleaned_fallback_dominant"
        primary_blocker = "Cleaned fallback still dominates the current hypothesis mix, so the first dry run should stay diagnostic."
    elif raw_source_count:
        source_mix = "raw_source_available"
        primary_blocker = "Raw separated sources are available, but the bridge still needs a tightly scoped first dry run."
    else:
        source_mix = "source_mix_unknown"
        primary_blocker = "The current source mix is unclear, so the next dry run should start by validating one exported case."

    return [
        {
            "bridge_status": str(readiness.get("bridge_status", "")),
            "source_mix": source_mix,
            "recommended_slice": "single_verified_case",
            "dry_run_goal": "Run one narrow diagnostic pass to validate the export path before any broader MeetEval or cpWER claim.",
            "primary_blocker": primary_blocker,
            "expected_evidence": "results/tables/meeteval_dry_run_receipt.json",
            "handoff_note": "MeetEval / cpWER has not been run yet; this card only frames the first dry run.",
        }
    ]


def build_meeteval_dry_run_handoff_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval Dry Run Handoff",
        "",
        "This generated handoff packet translates readiness into a single narrow next step. It does not claim that MeetEval or cpWER has been run.",
        "",
        "| bridge_status | source_mix | recommended_slice | dry_run_goal | primary_blocker | expected_evidence | handoff_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['bridge_status']} | {row['source_mix']} | {row['recommended_slice']} | {row['dry_run_goal']} | {row['primary_blocker']} | {row['expected_evidence']} | {row['handoff_note']} |"
        )
    return lines


def build_meeteval_dry_run_receipt_rows(handoff_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not handoff_rows:
        return []

    handoff = handoff_rows[0]
    return [
        {
            "execution_status": "template_only",
            "run_scope": str(handoff.get("recommended_slice", "")),
            "expected_inputs": "results/tables/meeteval_reference_segments.jsonl; results/tables/meeteval_hypothesis_segments.jsonl",
            "expected_outputs": "Diagnostic export-path confirmation and a narrow run note.",
            "writeback_note": "MeetEval / cpWER has not been executed yet; fill this receipt only after the first dry run.",
        }
    ]


def build_meeteval_dry_run_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval Dry Run Receipt",
        "",
        "This generated receipt is a template-only writeback target for the first narrow dry run. It does not claim that MeetEval or cpWER has been executed.",
        "",
        "| execution_status | run_scope | expected_inputs | expected_outputs | writeback_note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['run_scope']} | {row['expected_inputs']} | {row['expected_outputs']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    rows: list[dict[str, Any]],
    reference_lines: list[str],
    hypothesis_lines: list[str],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_compatibility_summary.csv"
    json_path = tables_dir / "meeteval_compatibility_summary.json"
    reference_path = tables_dir / "meeteval_reference_segments.jsonl"
    hypothesis_path = tables_dir / "meeteval_hypothesis_segments.jsonl"
    note_path = figures_dir / "meeteval_compatibility_note.md"
    readiness_rows = build_meeteval_readiness_rows(rows)
    readiness_csv_path = tables_dir / "meeteval_readiness.csv"
    readiness_json_path = tables_dir / "meeteval_readiness.json"
    readiness_note_path = figures_dir / "meeteval_readiness.md"
    dry_run_handoff_rows = build_meeteval_dry_run_handoff_rows(readiness_rows)
    dry_run_handoff_csv_path = tables_dir / "meeteval_dry_run_handoff.csv"
    dry_run_handoff_json_path = tables_dir / "meeteval_dry_run_handoff.json"
    dry_run_handoff_note_path = figures_dir / "meeteval_dry_run_handoff.md"
    dry_run_receipt_rows = build_meeteval_dry_run_receipt_rows(dry_run_handoff_rows)
    dry_run_receipt_json_path = tables_dir / "meeteval_dry_run_receipt.json"
    dry_run_receipt_note_path = figures_dir / "meeteval_dry_run_receipt.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    reference_path.write_text("\n".join(reference_lines) + ("\n" if reference_lines else ""), encoding="utf-8")
    hypothesis_path.write_text("\n".join(hypothesis_lines) + ("\n" if hypothesis_lines else ""), encoding="utf-8")
    note_path.write_text("\n".join(build_meeteval_compatibility_lines(rows)) + "\n", encoding="utf-8")
    with readiness_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=READINESS_COLUMNS)
        writer.writeheader()
        writer.writerows(readiness_rows)
    readiness_json_path.write_text(json.dumps(readiness_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness_note_path.write_text("\n".join(build_meeteval_readiness_lines(readiness_rows)) + "\n", encoding="utf-8")
    with dry_run_handoff_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DRY_RUN_HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerows(dry_run_handoff_rows)
    dry_run_handoff_json_path.write_text(json.dumps(dry_run_handoff_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    dry_run_handoff_note_path.write_text(
        "\n".join(build_meeteval_dry_run_handoff_lines(dry_run_handoff_rows)) + "\n",
        encoding="utf-8",
    )
    dry_run_receipt_json_path.write_text(json.dumps(dry_run_receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    dry_run_receipt_note_path.write_text(
        "\n".join(build_meeteval_dry_run_receipt_lines(dry_run_receipt_rows)) + "\n",
        encoding="utf-8",
    )
    return (
        csv_path,
        json_path,
        reference_path,
        hypothesis_path,
        note_path,
        readiness_csv_path,
        readiness_json_path,
        readiness_note_path,
        dry_run_handoff_csv_path,
        dry_run_handoff_json_path,
        dry_run_handoff_note_path,
        dry_run_receipt_json_path,
        dry_run_receipt_note_path,
    )


def main() -> None:
    case_ids = list_verified_cases()
    reference_payloads = {case_id: load_reference_payload(case_id) for case_id in case_ids}
    hypothesis_payloads = {case_id: load_hypothesis_payload(case_id) for case_id in case_ids}
    rows = build_meeteval_compatibility_rows(case_ids, reference_payloads, hypothesis_payloads)
    reference_lines: list[str] = []
    hypothesis_lines: list[str] = []
    for case_id in case_ids:
        reference_lines.extend(
            build_meeteval_segment_lines(case_id, "reference", list(reference_payloads[case_id].get("segments", [])))
        )
        hypothesis_lines.extend(
            build_meeteval_segment_lines(case_id, "hypothesis", list(hypothesis_payloads[case_id].get("segments", [])))
        )
    (
        csv_path,
        json_path,
        reference_path,
        hypothesis_path,
        note_path,
        readiness_csv_path,
        readiness_json_path,
        readiness_note_path,
        dry_run_handoff_csv_path,
        dry_run_handoff_json_path,
        dry_run_handoff_note_path,
        dry_run_receipt_json_path,
        dry_run_receipt_note_path,
    ) = write_outputs(rows, reference_lines, hypothesis_lines)
    print(f"Wrote MeetEval summary: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval reference export: {reference_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval hypothesis export: {hypothesis_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval note: {note_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval readiness CSV: {readiness_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval readiness JSON: {readiness_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval readiness note: {readiness_note_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run handoff CSV: {dry_run_handoff_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run handoff JSON: {dry_run_handoff_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run handoff note: {dry_run_handoff_note_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run receipt JSON: {dry_run_receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval dry run receipt note: {dry_run_receipt_note_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
