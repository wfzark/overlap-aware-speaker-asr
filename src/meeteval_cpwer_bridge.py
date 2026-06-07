from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .evaluate_cer import levenshtein_distance, normalize_text
from .meeteval_dry_run import load_jsonl_segments, select_preferred_case


BRIDGE_COLUMNS = [
    "case_id",
    "hypothesis_source",
    "speaker_count",
    "direct_macro_cer",
    "swapped_macro_cer",
    "cpwer_bridge_lite",
    "best_mapping",
    "observation",
]

HANDOFF_COLUMNS = [
    "bridge_status",
    "case_id",
    "cpwer_bridge_lite",
    "best_mapping",
    "bridge_goal",
    "primary_limitation",
    "expected_evidence",
    "handoff_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "run_scope",
    "case_id",
    "cpwer_bridge_lite",
    "best_mapping",
    "expected_inputs",
    "writeback_note",
]


def aggregate_speaker_text(segments: list[dict[str, Any]], speaker: str) -> str:
    texts: list[str] = []
    for segment in segments:
        if str(segment.get("speaker", "")).strip() == speaker:
            text = str(segment.get("text", "")).strip()
            if text:
                texts.append(text)
    return "".join(texts)


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    ref_norm = normalize_text(reference_text)
    hyp_norm = normalize_text(hypothesis_text)
    distance = levenshtein_distance(ref_norm, hyp_norm)
    reference_length = len(ref_norm)
    return round(distance / reference_length, 6) if reference_length else 0.0


def macro_cer_for_mapping(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
    speakers: list[str],
    mapping: dict[str, str],
) -> float:
    scores: list[float] = []
    for speaker in speakers:
        reference_text = aggregate_speaker_text(reference_segments, speaker)
        hypothesis_text = aggregate_speaker_text(hypothesis_segments, mapping[speaker])
        if reference_text:
            scores.append(compute_cer(reference_text, hypothesis_text))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


def build_cpwer_bridge_row(
    case_id: str,
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
    hypothesis_source: str = "",
) -> dict[str, Any]:
    speakers = sorted(
        {
            str(segment.get("speaker", "")).strip()
            for segment in reference_segments + hypothesis_segments
            if str(segment.get("speaker", "")).strip()
        }
    )
    if len(speakers) != 2:
        return {
            "case_id": case_id,
            "hypothesis_source": hypothesis_source,
            "speaker_count": len(speakers),
            "direct_macro_cer": 0.0,
            "swapped_macro_cer": 0.0,
            "cpwer_bridge_lite": 0.0,
            "best_mapping": "unsupported",
            "observation": "cpWER bridge-lite currently supports exactly two speakers per case.",
        }

    speaker_a, speaker_b = speakers
    direct_mapping = {speaker_a: speaker_a, speaker_b: speaker_b}
    swapped_mapping = {speaker_a: speaker_b, speaker_b: speaker_a}
    direct_macro_cer = macro_cer_for_mapping(
        reference_segments,
        hypothesis_segments,
        speakers,
        direct_mapping,
    )
    swapped_macro_cer = macro_cer_for_mapping(
        reference_segments,
        hypothesis_segments,
        speakers,
        swapped_mapping,
    )
    if swapped_macro_cer < direct_macro_cer:
        best_mapping = "swapped"
        cpwer_bridge_lite = swapped_macro_cer
    else:
        best_mapping = "direct"
        cpwer_bridge_lite = direct_macro_cer

    return {
        "case_id": case_id,
        "hypothesis_source": hypothesis_source,
        "speaker_count": len(speakers),
        "direct_macro_cer": direct_macro_cer,
        "swapped_macro_cer": swapped_macro_cer,
        "cpwer_bridge_lite": cpwer_bridge_lite,
        "best_mapping": best_mapping,
        "observation": (
            "experimental/frontier cpWER bridge-lite from JSONL exports; "
            "this is not a full MeetEval cpWER claim."
        ),
    }


def build_cpwer_bridge_lines(row: dict[str, Any]) -> list[str]:
    lines = [
        "# MeetEval cpWER Bridge",
        "",
        "This generated note records a narrow cpWER bridge-lite pass on exported segments. "
        "It does not claim a full MeetEval or official cpWER benchmark evaluation.",
        "",
        "| case_id | hypothesis_source | speaker_count | direct_macro_cer | swapped_macro_cer | cpwer_bridge_lite | best_mapping | observation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['case_id']} | {row['hypothesis_source']} | {row['speaker_count']} | "
            f"{row['direct_macro_cer']} | {row['swapped_macro_cer']} | {row['cpwer_bridge_lite']} | "
            f"{row['best_mapping']} | {row['observation']} |"
        ),
    ]
    return lines


def build_cpwer_bridge_handoff_rows(bridge_row: dict[str, Any]) -> list[dict[str, str]]:
    if not bridge_row:
        return []

    return [
        {
            "bridge_status": "cpwer_bridge_complete",
            "case_id": str(bridge_row.get("case_id", "")),
            "cpwer_bridge_lite": str(bridge_row.get("cpwer_bridge_lite", "")),
            "best_mapping": str(bridge_row.get("best_mapping", "")),
            "bridge_goal": "Use the bridge-lite result as a narrow compatibility signal before any broader MeetEval integration.",
            "primary_limitation": "This uses speaker-aggregated macro CER rather than a full MeetEval cpWER implementation.",
            "expected_evidence": "results/tables/meeteval_cpwer_bridge_receipt.json",
            "handoff_note": "MeetEval cpWER bridge-lite has been computed for one case; it is not a finished benchmark claim.",
        }
    ]


def build_cpwer_bridge_handoff_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Bridge Handoff",
        "",
        "This generated handoff packet turns the cpWER bridge-lite result into the next narrow frontier step.",
        "",
        "| bridge_status | case_id | cpwer_bridge_lite | best_mapping | bridge_goal | primary_limitation | expected_evidence | handoff_note |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['bridge_status']} | {row['case_id']} | {row['cpwer_bridge_lite']} | {row['best_mapping']} | "
            f"{row['bridge_goal']} | {row['primary_limitation']} | {row['expected_evidence']} | {row['handoff_note']} |"
        )
    return lines


def build_cpwer_bridge_receipt_rows(handoff_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not handoff_rows:
        return []

    handoff = handoff_rows[0]
    return [
        {
            "execution_status": "bridge_complete",
            "run_scope": "single_verified_case",
            "case_id": str(handoff.get("case_id", "")),
            "cpwer_bridge_lite": str(handoff.get("cpwer_bridge_lite", "")),
            "best_mapping": str(handoff.get("best_mapping", "")),
            "expected_inputs": "results/tables/meeteval_reference_segments.jsonl; results/tables/meeteval_hypothesis_segments.jsonl",
            "writeback_note": "cpWER bridge-lite complete for one case; full MeetEval evaluation remains pending.",
        }
    ]


def build_cpwer_bridge_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Bridge Receipt",
        "",
        "This receipt records the first cpWER bridge-lite writeback. It does not claim a finished MeetEval benchmark evaluation.",
        "",
        "| execution_status | run_scope | case_id | cpwer_bridge_lite | best_mapping | expected_inputs | writeback_note |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['run_scope']} | {row['case_id']} | {row['cpwer_bridge_lite']} | "
            f"{row['best_mapping']} | {row['expected_inputs']} | {row['writeback_note']} |"
        )
    return lines


def run_cpwer_bridge(case_id: str) -> dict[str, Any]:
    reference_path = PROJECT_ROOT / "results" / "tables" / "meeteval_reference_segments.jsonl"
    hypothesis_path = PROJECT_ROOT / "results" / "tables" / "meeteval_hypothesis_segments.jsonl"
    reference_segments = load_jsonl_segments(reference_path, case_id)
    hypothesis_segments = load_jsonl_segments(hypothesis_path, case_id)

    hypothesis_source = ""
    diagnostic_path = PROJECT_ROOT / "results" / "tables" / "meeteval_dry_run_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if str(diagnostic.get("case_id", "")) == case_id:
            hypothesis_source = str(diagnostic.get("hypothesis_source", ""))

    return build_cpwer_bridge_row(case_id, reference_segments, hypothesis_segments, hypothesis_source)


def write_outputs(
    bridge_row: dict[str, Any],
    handoff_rows: list[dict[str, str]],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bridge_csv_path = tables_dir / "meeteval_cpwer_bridge.csv"
    bridge_json_path = tables_dir / "meeteval_cpwer_bridge.json"
    bridge_md_path = figures_dir / "meeteval_cpwer_bridge.md"
    handoff_csv_path = tables_dir / "meeteval_cpwer_bridge_handoff.csv"
    handoff_json_path = tables_dir / "meeteval_cpwer_bridge_handoff.json"
    handoff_md_path = figures_dir / "meeteval_cpwer_bridge_handoff.md"
    receipt_json_path = tables_dir / "meeteval_cpwer_bridge_receipt.json"
    receipt_md_path = figures_dir / "meeteval_cpwer_bridge_receipt.md"

    with bridge_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_COLUMNS)
        writer.writeheader()
        writer.writerow(bridge_row)
    bridge_json_path.write_text(json.dumps(bridge_row, ensure_ascii=False, indent=2), encoding="utf-8")
    bridge_md_path.write_text("\n".join(build_cpwer_bridge_lines(bridge_row)) + "\n", encoding="utf-8")
    with handoff_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerows(handoff_rows)
    handoff_json_path.write_text(json.dumps(handoff_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    handoff_md_path.write_text("\n".join(build_cpwer_bridge_handoff_lines(handoff_rows)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_cpwer_bridge_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return (
        bridge_csv_path,
        bridge_json_path,
        bridge_md_path,
        handoff_csv_path,
        handoff_json_path,
        handoff_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def main() -> None:
    checklist_path = PROJECT_ROOT / "results" / "tables" / "meeteval_dry_run_checklist.csv"
    case_id = select_preferred_case(checklist_path)
    bridge_row = run_cpwer_bridge(case_id)
    handoff_rows = build_cpwer_bridge_handoff_rows(bridge_row)
    receipt_rows = build_cpwer_bridge_receipt_rows(handoff_rows)
    (
        bridge_csv_path,
        bridge_json_path,
        bridge_md_path,
        handoff_csv_path,
        handoff_json_path,
        handoff_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(bridge_row, handoff_rows, receipt_rows)
    print(f"Wrote MeetEval cpWER bridge CSV: {bridge_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge JSON: {bridge_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge note: {bridge_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge handoff CSV: {handoff_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge handoff JSON: {handoff_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge handoff note: {handoff_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval cpWER bridge receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")
    print(f"cpWER bridge-lite: {bridge_row['cpwer_bridge_lite']} ({bridge_row['best_mapping']})")


if __name__ == "__main__":
    main()
