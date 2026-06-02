from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, get_audio_cases, load_config


def find_case(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in get_audio_cases(config):
        if case["id"] == case_id:
            return case
    raise ValueError(f"Unknown audio case: {case_id}")


def read_transcript(case_id: str, speaker_index: int) -> dict[str, Any]:
    path = PROJECT_ROOT / "results" / "transcripts_raw" / f"{case_id}_spk{speaker_index}_whisper.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing transcript: {path.relative_to(PROJECT_ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def speaker_segments(transcript: dict[str, Any], speaker_label: str) -> list[dict[str, Any]]:
    return [
        {
            "speaker": speaker_label,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"].strip(),
        }
        for segment in transcript.get("segments", [])
        if segment.get("text", "").strip()
    ]


def build_full_text(segments: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{segment['speaker']}] {segment['text']}" for segment in segments)


def merge_case(case_id: str) -> Path:
    spk1 = read_transcript(case_id, 1)
    spk2 = read_transcript(case_id, 2)
    model = spk1.get("model", spk2.get("model", "unknown"))
    runtime_sec_total = round(
        float(spk1.get("runtime_sec", 0.0)) + float(spk2.get("runtime_sec", 0.0)),
        3,
    )
    segments = speaker_segments(spk1, "SPEAKER_1") + speaker_segments(spk2, "SPEAKER_2")
    segments.sort(key=lambda item: (item["start"], item["speaker"]))

    payload = {
        "case_id": case_id,
        "method": "separated_tracks_whisper",
        "model": model,
        "segments": segments,
        "full_text": build_full_text(segments),
        "runtime_sec_total": runtime_sec_total,
    }

    output_dir = PROJECT_ROOT / "results" / "transcripts_speaker"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_separated_speaker_transcript.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_separated_benchmark_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in get_audio_cases(config):
        case_id = case["id"]
        spk1_path = PROJECT_ROOT / "results" / "transcripts_raw" / f"{case_id}_spk1_whisper.json"
        spk2_path = PROJECT_ROOT / "results" / "transcripts_raw" / f"{case_id}_spk2_whisper.json"
        merged_path = PROJECT_ROOT / "results" / "transcripts_speaker" / f"{case_id}_separated_speaker_transcript.json"
        spk1 = json.loads(spk1_path.read_text(encoding="utf-8"))
        spk2 = json.loads(spk2_path.read_text(encoding="utf-8"))
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case_id": case_id,
                "overlap_level": case["overlap_level"],
                "model": merged.get("model", spk1.get("model", spk2.get("model", ""))),
                "spk1_audio_path": spk1.get("audio_path", ""),
                "spk2_audio_path": spk2.get("audio_path", ""),
                "spk1_runtime_sec": spk1.get("runtime_sec", 0.0),
                "spk2_runtime_sec": spk2.get("runtime_sec", 0.0),
                "runtime_sec_total": merged.get("runtime_sec_total", 0.0),
                "spk1_segments_count": len(spk1.get("segments", [])),
                "spk2_segments_count": len(spk2.get("segments", [])),
                "merged_segments_count": len(merged.get("segments", [])),
                "spk1_text_length": len(spk1.get("text", "")),
                "spk2_text_length": len(spk2.get("text", "")),
                "full_text_length": len(merged.get("full_text", "")),
                "speaker_transcript_path": merged_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    return rows


def write_separated_benchmark(config: dict[str, Any]) -> None:
    rows = build_separated_benchmark_rows(config)
    output_dir = PROJECT_ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "separated_asr_benchmark.csv"
    json_path = output_dir / "separated_asr_benchmark.json"
    csv_path.write_text("", encoding="utf-8")
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "overlap_level",
                "model",
                "spk1_audio_path",
                "spk2_audio_path",
                "spk1_runtime_sec",
                "spk2_runtime_sec",
                "runtime_sec_total",
                "spk1_segments_count",
                "spk2_segments_count",
                "merged_segments_count",
                "spk1_text_length",
                "spk2_text_length",
                "full_text_length",
                "speaker_transcript_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote separated ASR benchmark CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote separated ASR benchmark JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Benchmark rows: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge separated Whisper speaker tracks.")
    parser.add_argument("--case", required=True, help="Audio case id, e.g. NoOverlap or all")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing speaker transcript.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.case == "all":
        cases = get_audio_cases(config)
    else:
        cases = [find_case(config, args.case)]

    for case in cases:
        output_path = PROJECT_ROOT / "results" / "transcripts_speaker" / f"{case['id']}_separated_speaker_transcript.json"
        if output_path.exists() and not args.overwrite:
            print(f"skip existing transcript: {output_path.relative_to(PROJECT_ROOT)}")
            continue
        output = merge_case(case["id"])
        print(f"Wrote speaker transcript: {output.relative_to(PROJECT_ROOT)}")

    write_separated_benchmark(config)


if __name__ == "__main__":
    main()
