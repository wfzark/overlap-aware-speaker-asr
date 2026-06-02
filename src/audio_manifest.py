from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, get_audio_cases, load_config


MANIFEST_COLUMNS = [
    "case_id",
    "audio_type",
    "path",
    "duration_sec",
    "sample_rate",
    "channels",
    "overlap_level",
]


def read_wav_info(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.getnframes()
    return {
        "duration_sec": round(frames / sample_rate, 3) if sample_rate else 0.0,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def add_audio_row(
    rows: list[dict[str, Any]],
    case_id: str,
    audio_type: str,
    rel_path: Path,
    overlap_level: int,
) -> None:
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Missing audio file: {rel_path}")
    info = read_wav_info(abs_path)
    rows.append(
        {
            "case_id": case_id,
            "audio_type": audio_type,
            "path": rel_path.as_posix(),
            "duration_sec": info["duration_sec"],
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "overlap_level": overlap_level,
        }
    )


def build_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config["paths"]
    mixed_dir = Path(paths["mixed_audio_dir"])
    separated_dir = Path(paths["separated_audio_dir"])
    snippets_dir = Path(paths["snippets_dir"])
    rows: list[dict[str, Any]] = []

    for case in get_audio_cases(config):
        case_id = case["id"]
        overlap_level = int(case["overlap_level"])
        add_audio_row(rows, case_id, "mixed", mixed_dir / case["mixed"], overlap_level)
        add_audio_row(
            rows,
            case_id,
            "separated_spk1",
            separated_dir / case["separated"]["spk1"],
            overlap_level,
        )
        add_audio_row(
            rows,
            case_id,
            "separated_spk2",
            separated_dir / case["separated"]["spk2"],
            overlap_level,
        )

    for snippet in sorted((PROJECT_ROOT / snippets_dir).glob("*.wav")):
        rel_path = snippet.relative_to(PROJECT_ROOT)
        case_id = snippet.stem.split("_", 1)[0]
        add_audio_row(rows, case_id, "snippet", rel_path, -1)

    return rows


def write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_config()
    rows = build_manifest(config)
    output_path = PROJECT_ROOT / "results" / "tables" / "audio_manifest.csv"
    write_manifest(rows, output_path)
    print(f"Wrote {len(rows)} audio rows to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
