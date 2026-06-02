from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .build_synthetic_references import read_csv_rows, read_json, snippet_transcript_path
from .config import PROJECT_ROOT, load_config
from .evaluate_cer import compute_cer
from .postprocess_transcript import build_full_text, process_segments
from .transcribe_whisper import get_model_name, load_whisper_model, preview, transcribe_audio


RAW_DIR = PROJECT_ROOT / "results" / "synthetic_transcripts_raw"
SPEAKER_DIR = PROJECT_ROOT / "results" / "synthetic_transcripts_speaker"
CLEANED_DIR = PROJECT_ROOT / "results" / "synthetic_transcripts_postprocessed"
OUTPUT_COLUMNS = [
    "sample_id",
    "tier",
    "overlap_level",
    "method",
    "reference_path",
    "hypothesis_path",
    "reference_length",
    "hypothesis_length",
    "edit_distance",
    "cer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic overlap benchmark with silver references.")
    parser.add_argument("--case", required=True, help="Sample id or all")
    parser.add_argument("--model", choices=["tiny", "base", "small"], default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing synthetic transcripts.")
    return parser.parse_args()


def load_manifest_rows() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "results" / "tables" / "synthetic_manifest.csv"
    return read_csv_rows(path)


def select_rows(rows: list[dict[str, Any]], case_arg: str) -> list[dict[str, Any]]:
    if case_arg == "all":
        return rows
    matches = [row for row in rows if str(row.get("sample_id", "")) == case_arg]
    if not matches:
        raise ValueError(f"Unknown synthetic sample id: {case_arg}")
    return matches


def transcript_path(sample_id: str, suffix: str, directory: Path) -> Path:
    return directory / f"{sample_id}_{suffix}_whisper.json"


def speaker_transcript_path(sample_id: str) -> Path:
    return SPEAKER_DIR / f"{sample_id}_separated_speaker_transcript.json"


def cleaned_transcript_path(sample_id: str) -> Path:
    return CLEANED_DIR / f"{sample_id}_separated_speaker_transcript_cleaned.json"


def read_or_transcribe(
    model: Any,
    audio_path: Path,
    language: str,
    output_path: Path,
    payload_builder: callable,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        print(f"skip existing transcript: {output_path.relative_to(PROJECT_ROOT)}")
        return read_json(output_path)
    result = transcribe_audio(model, audio_path, language)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = payload_builder(result)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote transcript: {output_path.relative_to(PROJECT_ROOT)}")
    return payload


def build_raw_payload(sample_id: str, tier: str, audio_path: Path, language: str, model_name: str, result: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "tier": tier,
        "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
        "mode": mode,
        "model": f"whisper-{model_name}",
        "language": language,
        "text": result["text"],
        "segments": result["segments"],
        "runtime_sec": result["runtime_sec"],
    }


def build_speaker_payload(
    sample_id: str,
    tier: str,
    model_name: str,
    language: str,
    spk1: dict[str, Any],
    spk2: dict[str, Any],
) -> dict[str, Any]:
    segments = []
    segments.extend({"speaker": "SPEAKER_1", **segment} for segment in spk1.get("segments", []))
    segments.extend({"speaker": "SPEAKER_2", **segment} for segment in spk2.get("segments", []))
    segments.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)), str(item.get("speaker", ""))))
    full_text = build_full_text(segments)
    return {
        "sample_id": sample_id,
        "tier": tier,
        "method": "separated_tracks_whisper",
        "model": f"whisper-{model_name}",
        "language": language,
        "segments": segments,
        "full_text": full_text,
        "runtime_sec_total": round(float(spk1.get("runtime_sec", 0.0)) + float(spk2.get("runtime_sec", 0.0)), 3),
        "spk1_runtime_sec": spk1.get("runtime_sec", 0.0),
        "spk2_runtime_sec": spk2.get("runtime_sec", 0.0),
    }


def build_cleaned_payload(sample_id: str, tier: str, speaker_payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    cleaned_segments, removed_segments = process_segments(list(speaker_payload.get("segments", [])))
    cleaned_full_text = build_full_text(cleaned_segments)
    return {
        "sample_id": sample_id,
        "tier": tier,
        "method": "duplicate_suppression",
        "source_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "cleaned_segments": cleaned_segments,
        "cleaned_full_text": cleaned_full_text,
        "removed_segments": removed_segments,
        "removed_count": len(removed_segments),
    }


def write_transcript(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_sample(
    row: dict[str, Any],
    model: Any,
    model_name: str,
    language: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    sample_id = str(row["sample_id"])
    tier = str(row["tier"])
    reference_path = PROJECT_ROOT / str(row["reference_path"])
    reference = read_json(reference_path)
    reference_text = str(reference.get("full_text", ""))

    mixed_audio = PROJECT_ROOT / str(row["mixed_path"])
    spk1_audio = PROJECT_ROOT / str(row["spk1_path"])
    spk2_audio = PROJECT_ROOT / str(row["spk2_path"])

    mixed_output = transcript_path(sample_id, "mixed", RAW_DIR)
    spk1_output = transcript_path(sample_id, "spk1", RAW_DIR)
    spk2_output = transcript_path(sample_id, "spk2", RAW_DIR)
    speaker_output = speaker_transcript_path(sample_id)
    cleaned_output = cleaned_transcript_path(sample_id)

    mixed_payload = read_or_transcribe(
        model,
        mixed_audio,
        language,
        mixed_output,
        lambda result: build_raw_payload(sample_id, tier, mixed_audio, language, model_name, result, "mixed"),
        overwrite,
    )

    spk1_payload = read_or_transcribe(
        model,
        spk1_audio,
        language,
        spk1_output,
        lambda result: build_raw_payload(sample_id, tier, spk1_audio, language, model_name, result, "separated_spk1"),
        overwrite,
    )
    spk2_payload = read_or_transcribe(
        model,
        spk2_audio,
        language,
        spk2_output,
        lambda result: build_raw_payload(sample_id, tier, spk2_audio, language, model_name, result, "separated_spk2"),
        overwrite,
    )

    if speaker_output.exists() and not overwrite:
        print(f"skip existing speaker transcript: {speaker_output.relative_to(PROJECT_ROOT)}")
        speaker_payload = read_json(speaker_output)
    else:
        speaker_payload = build_speaker_payload(sample_id, tier, model_name, language, spk1_payload, spk2_payload)
        write_transcript(speaker_output, speaker_payload)
        print(f"Wrote speaker transcript: {speaker_output.relative_to(PROJECT_ROOT)}")

    if cleaned_output.exists() and not overwrite:
        print(f"skip existing cleaned transcript: {cleaned_output.relative_to(PROJECT_ROOT)}")
        cleaned_payload = read_json(cleaned_output)
    else:
        cleaned_payload = build_cleaned_payload(sample_id, tier, speaker_payload, speaker_output)
        write_transcript(cleaned_output, cleaned_payload)
        print(f"Wrote cleaned transcript: {cleaned_output.relative_to(PROJECT_ROOT)}")

    rows = [
        {
            "sample_id": sample_id,
            "tier": tier,
            "overlap_level": row.get("overlap_level_numeric", row.get("overlap_level", "")),
            "method": "mixed_whisper",
            "reference_path": reference_path.relative_to(PROJECT_ROOT).as_posix(),
            "hypothesis_path": mixed_output.relative_to(PROJECT_ROOT).as_posix(),
            "reference_length": compute_cer(reference_text, str(mixed_payload.get("text", "")))["reference_length"],
            "hypothesis_length": compute_cer(reference_text, str(mixed_payload.get("text", "")))["hypothesis_length"],
            "edit_distance": compute_cer(reference_text, str(mixed_payload.get("text", "")))["edit_distance"],
            "cer": compute_cer(reference_text, str(mixed_payload.get("text", "")))["cer"],
        },
        {
            "sample_id": sample_id,
            "tier": tier,
            "overlap_level": row.get("overlap_level_numeric", row.get("overlap_level", "")),
            "method": "separated_whisper",
            "reference_path": reference_path.relative_to(PROJECT_ROOT).as_posix(),
            "hypothesis_path": speaker_output.relative_to(PROJECT_ROOT).as_posix(),
            "reference_length": compute_cer(reference_text, str(speaker_payload.get("full_text", "")))["reference_length"],
            "hypothesis_length": compute_cer(reference_text, str(speaker_payload.get("full_text", "")))["hypothesis_length"],
            "edit_distance": compute_cer(reference_text, str(speaker_payload.get("full_text", "")))["edit_distance"],
            "cer": compute_cer(reference_text, str(speaker_payload.get("full_text", "")))["cer"],
        },
        {
            "sample_id": sample_id,
            "tier": tier,
            "overlap_level": row.get("overlap_level_numeric", row.get("overlap_level", "")),
            "method": "separated_whisper_cleaned",
            "reference_path": reference_path.relative_to(PROJECT_ROOT).as_posix(),
            "hypothesis_path": cleaned_output.relative_to(PROJECT_ROOT).as_posix(),
            "reference_length": compute_cer(reference_text, str(cleaned_payload.get("cleaned_full_text", "")))["reference_length"],
            "hypothesis_length": compute_cer(reference_text, str(cleaned_payload.get("cleaned_full_text", "")))["hypothesis_length"],
            "edit_distance": compute_cer(reference_text, str(cleaned_payload.get("cleaned_full_text", "")))["edit_distance"],
            "cer": compute_cer(reference_text, str(cleaned_payload.get("cleaned_full_text", "")))["cer"],
        },
    ]
    for item in rows:
        item["cer"] = float(item["cer"])
    return rows


def main() -> None:
    args = parse_args()
    config = load_config()
    model_name = get_model_name(config, args.model)
    language = config.get("asr", {}).get("language", "zh")
    rows = select_rows(load_manifest_rows(), args.case)
    model = load_whisper_model(model_name)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        output_rows.extend(evaluate_sample(row, model, model_name, language, args.overwrite))

    output_dir = PROJECT_ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_cer_results.csv"
    json_path = output_dir / "synthetic_cer_results.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    json_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote synthetic CER CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote synthetic CER JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Synthetic CER rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
