from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .adaptive_router import select_method
from .build_synthetic_references import read_csv_rows, read_json
from .config import PROJECT_ROOT


DECISION_COLUMNS = [
    "sample_id",
    "tier",
    "overlap_level",
    "selected_method",
    "decision_rule",
    "mixed_segments_count",
    "separated_segments_count",
    "cleaned_segments_count",
    "mixed_text_length",
    "separated_text_length",
    "cleaned_text_length",
    "text_length_ratio",
    "mixed_runtime_sec",
    "separated_runtime_sec",
    "cleaned_runtime_sec",
    "runtime_ratio",
    "duplicate_removed_count",
    "notes",
]
PERFORMANCE_COLUMNS = ["tier", "strategy", "average_cer", "sample_count"]
STRATEGIES = [
    "fixed_mixed_whisper",
    "fixed_separated_whisper",
    "fixed_separated_whisper_cleaned",
    "oracle_best",
    "rule_router",
]
TIER_TO_LEVEL = {
    "SyntheticNoOverlap": 0,
    "SyntheticLightOverlap": 1,
    "SyntheticMidOverlap": 2,
    "SyntheticHeavyOverlap": 3,
    "SyntheticOppositeOverlap": 4,
}


def to_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def to_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if isinstance(row, dict)]


def load_manifest() -> list[dict[str, Any]]:
    return read_csv(PROJECT_ROOT / "results" / "tables" / "synthetic_manifest.csv")


def load_cer_lookup() -> dict[tuple[str, str], float]:
    rows = read_csv(PROJECT_ROOT / "results" / "tables" / "synthetic_cer_results.csv")
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        method = str(row.get("method", "")).strip()
        if not sample_id or not method:
            continue
        lookup[(sample_id, method)] = to_float(row.get("cer"))
    return lookup


def load_cleaned_payloads() -> dict[str, dict[str, Any]]:
    cleaned_dir = PROJECT_ROOT / "results" / "synthetic_transcripts_postprocessed"
    payloads: dict[str, dict[str, Any]] = {}
    if not cleaned_dir.exists():
        return payloads
    for path in cleaned_dir.glob("*_separated_speaker_transcript_cleaned.json"):
        try:
            payload = read_json(path)
        except Exception as exc:
            print(f"warning: failed to read cleaned transcript {path.relative_to(PROJECT_ROOT)}: {exc}")
            continue
        sample_id = str(payload.get("sample_id", "")).strip()
        if sample_id:
            payloads[sample_id] = payload
    return payloads


def load_transcript_payloads(sample_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mixed_path = PROJECT_ROOT / "results" / "synthetic_transcripts_raw" / f"{sample_id}_mixed_whisper.json"
    speaker_path = PROJECT_ROOT / "results" / "synthetic_transcripts_speaker" / f"{sample_id}_separated_speaker_transcript.json"
    cleaned_path = PROJECT_ROOT / "results" / "synthetic_transcripts_postprocessed" / f"{sample_id}_separated_speaker_transcript_cleaned.json"
    mixed = read_json(mixed_path)
    separated = read_json(speaker_path)
    cleaned = read_json(cleaned_path)
    return mixed, separated, cleaned


def build_decisions(
    manifest_rows: list[dict[str, Any]],
    cleaned_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        tier = str(row.get("tier", "")).strip()
        level = TIER_TO_LEVEL.get(tier, to_int(row.get("overlap_level_numeric", 0)))
        selected_method, decision_rule = select_method(level)
        mixed, separated, cleaned = load_transcript_payloads(sample_id)
        cleaned_payload = cleaned_rows.get(sample_id, cleaned)

        mixed_segments = mixed.get("segments", [])
        separated_segments = separated.get("segments", [])
        cleaned_segments = cleaned.get("cleaned_segments", [])
        mixed_text_length = len(str(mixed.get("text", "")))
        separated_text_length = len(str(separated.get("full_text", "")))
        cleaned_text_length = len(str(cleaned.get("cleaned_full_text", "")))
        mixed_runtime_sec = to_float(mixed.get("runtime_sec"))
        separated_runtime_sec = to_float(separated.get("runtime_sec_total"))
        cleaned_runtime_sec = separated_runtime_sec
        runtime_ratio = round(separated_runtime_sec / mixed_runtime_sec, 6) if mixed_runtime_sec else 0.0
        duplicate_removed_count = to_int(cleaned_payload.get("removed_count", cleaned.get("removed_count", 0)))
        notes = "Router is rule-based and does not use CER as an input feature."
        if cleaned_segments:
            notes += " Cleaned transcript is retained as a fallback candidate but not selected by the initial rule."

        decisions.append(
            {
                "sample_id": sample_id,
                "tier": tier,
                "overlap_level": level,
                "selected_method": selected_method,
                "decision_rule": decision_rule,
                "mixed_segments_count": len(mixed_segments),
                "separated_segments_count": len(separated_segments),
                "cleaned_segments_count": len(cleaned_segments),
                "mixed_text_length": mixed_text_length,
                "separated_text_length": separated_text_length,
                "cleaned_text_length": cleaned_text_length,
                "text_length_ratio": round(separated_text_length / mixed_text_length, 6) if mixed_text_length else 0.0,
                "mixed_runtime_sec": mixed_runtime_sec,
                "separated_runtime_sec": separated_runtime_sec,
                "cleaned_runtime_sec": cleaned_runtime_sec,
                "runtime_ratio": runtime_ratio,
                "duplicate_removed_count": duplicate_removed_count,
                "notes": notes,
            }
        )
    return decisions


def compute_strategy_averages(
    cer_lookup: dict[tuple[str, str], float],
    manifest_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sample_ids = [str(row.get("sample_id", "")).strip() for row in manifest_rows if str(row.get("sample_id", "")).strip()]
    tiers = sorted({str(row.get("tier", "")).strip() for row in manifest_rows if str(row.get("tier", "")).strip()})
    results: list[dict[str, Any]] = []

    def append_rows(scope: str, ids: list[str]) -> None:
        buckets: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}
        for sample_id in ids:
            mixed = cer_lookup.get((sample_id, "mixed_whisper"))
            separated = cer_lookup.get((sample_id, "separated_whisper"))
            cleaned = cer_lookup.get((sample_id, "separated_whisper_cleaned"))
            available = [v for v in [mixed, separated, cleaned] if v is not None]
            if mixed is not None:
                buckets["fixed_mixed_whisper"].append(mixed)
            if separated is not None:
                buckets["fixed_separated_whisper"].append(separated)
            if cleaned is not None:
                buckets["fixed_separated_whisper_cleaned"].append(cleaned)
            if available:
                buckets["oracle_best"].append(min(available))

        decision_lookup = {row["sample_id"]: row["selected_method"] for row in decisions}
        for sample_id in ids:
            method = decision_lookup.get(sample_id)
            if method is not None:
                cer = cer_lookup.get((sample_id, method))
                if cer is not None:
                    buckets["rule_router"].append(cer)

        for strategy, values in buckets.items():
            results.append(
                {
                    "tier": scope,
                    "strategy": strategy,
                    "average_cer": round(sum(values) / len(values), 6) if values else 0.0,
                    "sample_count": len(values),
                }
            )

    append_rows("ALL", sample_ids)
    for tier in tiers:
        tier_ids = [str(row.get("sample_id", "")).strip() for row in manifest_rows if str(row.get("tier", "")).strip() == tier]
        append_rows(tier, tier_ids)
    return results


def write_outputs(decisions: list[dict[str, Any]], performance: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    table_dir = PROJECT_ROOT / "results" / "tables"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    decisions_csv = table_dir / "synthetic_routing_decisions.csv"
    performance_csv = table_dir / "synthetic_routing_performance.csv"
    summary_md = fig_dir / "synthetic_routing_summary.md"

    with decisions_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        writer.writerows(decisions)

    with performance_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PERFORMANCE_COLUMNS)
        writer.writeheader()
        writer.writerows(performance)

    lines: list[str] = []
    lines.append("# Synthetic Routing Stability Summary")
    lines.append("")
    lines.append(
        "This is a stability check based on silver references built from snippet Whisper transcriptions. "
        "It complements, but does not replace, the five manually verified gold benchmark cases."
    )
    lines.append("")
    lines.append("## Overall Average CER")
    lines.append("")
    lines.append("| strategy | average_cer | sample_count |")
    lines.append("| --- | ---: | ---: |")
    for row in [r for r in performance if r["tier"] == "ALL"]:
        lines.append(f"| {row['strategy']} | {row['average_cer']} | {row['sample_count']} |")
    lines.append("")
    lines.append("## Tier Breakdown")
    lines.append("")
    for tier in sorted({row["tier"] for row in performance if row["tier"] != "ALL"}):
        lines.append(f"### {tier}")
        lines.append("")
        lines.append("| strategy | average_cer | sample_count |")
        lines.append("| --- | ---: | ---: |")
        tier_rows = [row for row in performance if row["tier"] == tier]
        for row in tier_rows:
            lines.append(f"| {row['strategy']} | {row['average_cer']} | {row['sample_count']} |")
        lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return decisions_csv, performance_csv, summary_md


def main() -> None:
    manifest_rows = load_manifest()
    cleaned_rows = load_cleaned_payloads()
    decisions = build_decisions(manifest_rows, cleaned_rows)
    cer_lookup = load_cer_lookup()
    performance = compute_strategy_averages(cer_lookup, manifest_rows, decisions)
    decisions_csv, performance_csv, summary_md = write_outputs(decisions, performance)

    decisions_json = decisions_csv.with_suffix(".json")
    performance_json = performance_csv.with_suffix(".json")
    decisions_json.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    performance_json.write_text(json.dumps(performance, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote synthetic routing decisions: {decisions_csv.relative_to(PROJECT_ROOT)}")
    print(f"Wrote synthetic routing performance: {performance_csv.relative_to(PROJECT_ROOT)}")
    print(f"Wrote synthetic routing summary: {summary_md.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
