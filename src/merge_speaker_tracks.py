from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge separated Whisper speaker tracks.")
    parser.add_argument("--case", required=True, help="Audio case id, e.g. NoOverlap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    find_case(config, args.case)
    output_path = merge_case(args.case)
    print(f"Wrote speaker transcript: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
