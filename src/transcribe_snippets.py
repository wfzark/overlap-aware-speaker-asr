from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config
from .transcribe_whisper import load_whisper_model, preview, transcribe_audio, get_model_name


SNIPPET_COLUMNS = [
    "snippet_id",
    "source_group",
    "audio_path",
    "model",
    "runtime_sec",
    "segments_count",
    "text_length",
    "text_preview",
    "transcript_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe snippet audio files with lightweight Whisper.")
    parser.add_argument("--model", choices=["tiny", "base", "small"], default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing snippet transcripts.")
    return parser.parse_args()


def snippet_transcript_path(snippet_id: str) -> Path:
    return PROJECT_ROOT / "results" / "snippet_transcripts" / f"{snippet_id}_whisper.json"


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if isinstance(row, dict)]


def write_snippet_transcript(
    snippet_id: str,
    audio_path: Path,
    model_name: str,
    language: str,
    result: dict[str, Any],
) -> Path:
    output_path = snippet_transcript_path(snippet_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snippet_id": snippet_id,
        "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix(),
        "model": f"whisper-{model_name}",
        "language": language,
        "text": result["text"],
        "segments": result["segments"],
        "runtime_sec": result["runtime_sec"],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config()
    model_name = get_model_name(config, args.model)
    language = config.get("asr", {}).get("language", "zh")
    snippets_dir = PROJECT_ROOT / config["paths"]["snippets_dir"]
    wav_paths = sorted(snippets_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No snippet wav files found in {snippets_dir.relative_to(PROJECT_ROOT)}")

    model = load_whisper_model(model_name)
    rows: list[dict[str, Any]] = []

    for audio_path in wav_paths:
        snippet_id = audio_path.stem
        output_path = snippet_transcript_path(snippet_id)
        if output_path.exists() and not args.overwrite:
            print(f"skip existing snippet transcript: {output_path.relative_to(PROJECT_ROOT)}")
            transcript = json.loads(output_path.read_text(encoding="utf-8-sig"))
        else:
            result = transcribe_audio(model, audio_path, language)
            output_path = write_snippet_transcript(snippet_id, audio_path, model_name, language, result)
            transcript = json.loads(output_path.read_text(encoding="utf-8-sig"))
            print(f"Wrote snippet transcript: {output_path.relative_to(PROJECT_ROOT)}")

        text = str(transcript.get("text", ""))
        rows.append(
            {
                "snippet_id": snippet_id,
                "source_group": snippet_id.split("_", 1)[0],
                "audio_path": transcript.get("audio_path", ""),
                "model": transcript.get("model", ""),
                "runtime_sec": transcript.get("runtime_sec", 0.0),
                "segments_count": len(transcript.get("segments", [])),
                "text_length": len(text),
                "text_preview": preview(text),
                "transcript_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    table_path = PROJECT_ROOT / "results" / "tables" / "snippet_transcripts.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SNIPPET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote snippet transcript table: {table_path.relative_to(PROJECT_ROOT)}")
    print(f"Snippet rows: {len(rows)}")


if __name__ == "__main__":
    main()
