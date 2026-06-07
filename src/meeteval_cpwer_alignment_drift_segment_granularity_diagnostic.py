from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .meeteval_dry_run import load_jsonl_segments


SPEAKER_COLUMNS = [
    "case_id",
    "speaker",
    "reference_segment_count",
    "hypothesis_segment_count",
    "reference_avg_segment_duration_sec",
    "hypothesis_avg_segment_duration_sec",
    "avg_duration_delta_sec",
    "granularity_match",
]

SUMMARY_COLUMNS = [
    "case_id",
    "mismatched_speaker_count",
    "speaker_granularity_match",
    "dominant_blocker",
    "diagnostic_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "diagnostic_scope",
    "case_id",
    "mismatched_speaker_count",
    "writeback_note",
]


def load_timing_handoff() -> dict[str, Any]:
    handoff_path = (
        PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_alignment_drift_segment_timing_diagnostic_handoff.json"
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


def sum_duration_per_speaker(segments: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for segment in segments:
        speaker = str(segment.get("speaker", "")).strip()
        if not speaker:
            continue
        start = float(segment.get("start_time", 0.0) or 0.0)
        end = float(segment.get("end_time", 0.0) or 0.0)
        totals[speaker] += max(0.0, end - start)
    return dict(totals)


def avg_duration(count: int, total: float) -> float:
    if count <= 0:
        return 0.0
    return total / count


def build_speaker_rows(
    case_id: str,
    reference_counts: Counter[str],
    hypothesis_counts: Counter[str],
    reference_durations: dict[str, float],
    hypothesis_durations: dict[str, float],
    tolerance_sec: float = 0.05,
) -> list[dict[str, str]]:
    speakers = sorted(set(reference_counts) | set(hypothesis_counts))
    rows: list[dict[str, str]] = []
    for speaker in speakers:
        reference_count = reference_counts.get(speaker, 0)
        hypothesis_count = hypothesis_counts.get(speaker, 0)
        reference_avg = avg_duration(reference_count, reference_durations.get(speaker, 0.0))
        hypothesis_avg = avg_duration(hypothesis_count, hypothesis_durations.get(speaker, 0.0))
        delta = hypothesis_avg - reference_avg
        granularity_match = abs(delta) <= tolerance_sec
        rows.append(
            {
                "case_id": case_id,
                "speaker": speaker,
                "reference_segment_count": str(reference_count),
                "hypothesis_segment_count": str(hypothesis_count),
                "reference_avg_segment_duration_sec": f"{reference_avg:.3f}",
                "hypothesis_avg_segment_duration_sec": f"{hypothesis_avg:.3f}",
                "avg_duration_delta_sec": f"{delta:.3f}",
                "granularity_match": str(granularity_match),
            }
        )
    return rows


def build_summary_row(case_id: str, speaker_rows: list[dict[str, str]]) -> dict[str, str]:
    mismatched = [row for row in speaker_rows if row["granularity_match"] == "False"]
    mismatched_speaker_count = len(mismatched)
    speaker_granularity_match = mismatched_speaker_count == 0
    if mismatched:
        dominant = max(mismatched, key=lambda row: abs(float(row["avg_duration_delta_sec"])))
        dominant_blocker = f"{dominant['speaker']} delta={dominant['avg_duration_delta_sec']}s"
        diagnostic_note = (
            f"Per-speaker segment granularity drift detected for {case_id}; "
            f"mismatched_speaker_count={mismatched_speaker_count}. "
            "Reconciled alignment and cpWER execution remain pending."
        )
    else:
        dominant_blocker = "none"
        diagnostic_note = (
            f"Per-speaker segment granularity aligns for {case_id}; "
            "reconciled alignment and cpWER execution remain pending."
        )
    return {
        "case_id": case_id,
        "mismatched_speaker_count": str(mismatched_speaker_count),
        "speaker_granularity_match": str(speaker_granularity_match),
        "dominant_blocker": dominant_blocker,
        "diagnostic_note": diagnostic_note,
    }


def build_speaker_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Granularity Diagnostic",
        "",
        "This generated note records per-speaker average segment duration comparison for the drift case. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| case_id | speaker | reference_segment_count | hypothesis_segment_count | reference_avg_segment_duration_sec | hypothesis_avg_segment_duration_sec | avg_duration_delta_sec | granularity_match |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['speaker']} | {row['reference_segment_count']} | "
            f"{row['hypothesis_segment_count']} | {row['reference_avg_segment_duration_sec']} | "
            f"{row['hypothesis_avg_segment_duration_sec']} | {row['avg_duration_delta_sec']} | "
            f"{row['granularity_match']} |"
        )
    return lines


def build_summary_lines(summary: dict[str, str]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Granularity Diagnostic Summary",
        "",
        "This generated summary condenses the per-speaker segment granularity drift for the reconciliation case. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| case_id | mismatched_speaker_count | speaker_granularity_match | dominant_blocker | diagnostic_note |",
        "| --- | ---: | --- | --- | --- |",
        (
            f"| {summary['case_id']} | {summary['mismatched_speaker_count']} | "
            f"{summary['speaker_granularity_match']} | {summary['dominant_blocker']} | {summary['diagnostic_note']} |"
        ),
    ]
    return lines


def build_receipt_row(summary: dict[str, str]) -> dict[str, str]:
    return {
        "execution_status": "granularity_diagnostic_complete",
        "diagnostic_scope": "single_drift_case_per_speaker_granularity",
        "case_id": str(summary.get("case_id", "")),
        "mismatched_speaker_count": str(summary.get("mismatched_speaker_count", "0")),
        "writeback_note": (
            "Per-speaker segment granularity diagnostic complete for the drift case. "
            "Reconciled alignment and cpWER execution remain pending."
        ),
    }


def build_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Alignment Drift Segment Granularity Diagnostic Receipt",
        "",
        "This receipt records the per-speaker granularity diagnostic writeback. "
        "It does not claim reconciled alignment or cpWER execution.",
        "",
        "| execution_status | diagnostic_scope | case_id | mismatched_speaker_count | writeback_note |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['diagnostic_scope']} | {row['case_id']} | "
            f"{row['mismatched_speaker_count']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    speaker_rows: list[dict[str, str]],
    summary: dict[str, str],
    receipt_row: dict[str, str],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    speaker_csv_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic.csv"
    speaker_json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic.json"
    speaker_md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic.md"
    summary_csv_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_summary.csv"
    summary_json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_summary.json"
    summary_md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_summary.md"
    receipt_json_path = tables_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_receipt.json"
    receipt_md_path = figures_dir / "meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_receipt.md"

    with speaker_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SPEAKER_COLUMNS)
        writer.writeheader()
        writer.writerows(speaker_rows)
    speaker_json_path.write_text(json.dumps(speaker_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    speaker_md_path.write_text("\n".join(build_speaker_lines(speaker_rows)) + "\n", encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(summary)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text("\n".join(build_summary_lines(summary)) + "\n", encoding="utf-8")

    receipt_json_path.write_text(json.dumps([receipt_row], ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_receipt_lines([receipt_row])) + "\n", encoding="utf-8")
    return (
        speaker_csv_path,
        speaker_json_path,
        speaker_md_path,
        summary_csv_path,
        summary_json_path,
        summary_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def run_granularity_diagnostic(case_id: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    reference_path = PROJECT_ROOT / "results" / "tables" / "meeteval_reference_segments.jsonl"
    hypothesis_path = PROJECT_ROOT / "results" / "tables" / "meeteval_hypothesis_segments.jsonl"
    reference_segments = load_jsonl_segments(reference_path, case_id)
    hypothesis_segments = load_jsonl_segments(hypothesis_path, case_id)
    reference_counts = count_segments_per_speaker(reference_segments)
    hypothesis_counts = count_segments_per_speaker(hypothesis_segments)
    speaker_rows = build_speaker_rows(
        case_id,
        reference_counts,
        hypothesis_counts,
        sum_duration_per_speaker(reference_segments),
        sum_duration_per_speaker(hypothesis_segments),
    )
    summary = build_summary_row(case_id, speaker_rows)
    return speaker_rows, summary


def main() -> None:
    handoff = load_timing_handoff()
    case_id = str(handoff.get("case_id", "HeavyOverlap"))
    speaker_rows, summary = run_granularity_diagnostic(case_id)
    receipt_row = build_receipt_row(summary)
    (
        speaker_csv_path,
        speaker_json_path,
        speaker_md_path,
        summary_csv_path,
        summary_json_path,
        summary_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(speaker_rows, summary, receipt_row)
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic CSV: "
        f"{speaker_csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic JSON: "
        f"{speaker_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic note: "
        f"{speaker_md_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic summary CSV: "
        f"{summary_csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic summary JSON: "
        f"{summary_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic summary note: "
        f"{summary_md_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic receipt JSON: "
        f"{receipt_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER alignment drift segment granularity diagnostic receipt note: "
        f"{receipt_md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Mismatched speaker count: {summary['mismatched_speaker_count']}")


if __name__ == "__main__":
    main()
