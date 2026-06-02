from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config


def load_reference_cases() -> dict[str, dict[str, Any]]:
    ref_path = PROJECT_ROOT / "references" / "reference_transcripts.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference file: {ref_path.relative_to(PROJECT_ROOT)}")
    return json.loads(ref_path.read_text(encoding="utf-8-sig"))


def get_cases_to_bundle(case_arg: str) -> list[dict[str, Any]]:
    config = load_config()
    cases = list(config.get("audio_cases", []))
    refs = load_reference_cases()
    verified = {
        case_id
        for case_id, payload in refs.items()
        if payload.get("status") == "verified_reference"
    }

    if case_arg == "all_remaining":
        return [case for case in cases if case.get("id") not in verified]

    selected = [case for case in cases if case.get("id") == case_arg]
    if not selected:
        raise KeyError(f"Case '{case_arg}' not found in configs/config.yaml")
    return selected


def required_files_for(case_id: str, config: dict[str, Any]) -> list[Path]:
    paths_cfg = config.get("paths", {})
    mixed_dir = PROJECT_ROOT / paths_cfg.get("mixed_audio_dir", "resources/mixed_audio")
    separated_dir = PROJECT_ROOT / paths_cfg.get("separated_audio_dir", "resources/separated_audio")
    results_dir = PROJECT_ROOT / paths_cfg.get("results_dir", "results")

    return [
        mixed_dir / f"{case_id}.wav",
        separated_dir / f"{case_id}_spk1.wav",
        separated_dir / f"{case_id}_spk2.wav",
        results_dir / "transcripts_raw" / f"{case_id}_mixed_whisper.json",
        results_dir / "transcripts_raw" / f"{case_id}_spk1_whisper.json",
        results_dir / "transcripts_raw" / f"{case_id}_spk2_whisper.json",
        results_dir / "transcripts_speaker" / f"{case_id}_separated_speaker_transcript.json",
    ]


def build_bundle(case_id: str, config: dict[str, Any]) -> tuple[Path, list[Path]]:
    bundle_root = PROJECT_ROOT / "chat_upload" / case_id
    bundle_root.mkdir(parents=True, exist_ok=True)
    output_zip = PROJECT_ROOT / "chat_upload" / f"{case_id}_bundle.zip"

    files = required_files_for(case_id, config)
    missing = [path for path in files if not path.exists()]
    if missing:
        print(f"[{case_id}] missing files:")
        for path in missing:
            print(f"  - {path.relative_to(PROJECT_ROOT)}")
        return output_zip, missing

    copied_paths: list[Path] = []
    for src in files:
        dst = bundle_root / src.name
        shutil.copy2(src, dst)
        copied_paths.append(dst)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in copied_paths:
            zf.write(file_path, arcname=file_path.name)

    print(f"Wrote bundle: {output_zip.relative_to(PROJECT_ROOT)}")
    print(f"Bundle folder: {bundle_root.relative_to(PROJECT_ROOT)}")
    print(f"File count: {len(copied_paths)}")
    return output_zip, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare upload bundles for reference drafting.")
    parser.add_argument("--case", required=True, help="Case id or all_remaining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    cases = get_cases_to_bundle(args.case)
    if not cases:
        print("No remaining unverified cases to bundle.")
        return

    for case in cases:
        build_bundle(case["id"], config)


if __name__ == "__main__":
    main()
