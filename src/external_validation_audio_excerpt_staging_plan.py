from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config
from .external_validation_slice_scaffold import MAPPING_COLUMNS, build_mapping_row


PLAN_COLUMNS = [
    "dataset_name",
    "slice_id",
    "label",
    "license_status",
    "audio_path",
    "reference_path",
    "audio_staged",
    "reference_template_staged",
    "staging_status",
    "download_source",
    "result_label",
    "staging_note",
]

SUMMARY_COLUMNS = [
    "metric",
    "value",
    "label",
]

AISHELL4_DOWNLOAD_URL = "https://www.openslr.org/111/"


def load_slice_mapping() -> dict[str, Any]:
    mapping_path = PROJECT_ROOT / "results" / "tables" / "external_validation_slice_mapping.json"
    if not mapping_path.exists():
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_reference_template(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "slice_id": str(mapping.get("slice_id", "")),
        "dataset_name": str(mapping.get("dataset_name", "AISHELL-4")),
        "label": str(mapping.get("label", "external/sanity-check")),
        "license_id": str(mapping.get("license_id", "CC BY-SA 4.0")),
        "speaker_schema": mapping.get("speaker_schema", {}),
        "segments": [],
        "staging_status": "reference_template_only",
        "staging_note": (
            "Reference template for the first external slice. "
            "Populate segments after a licensed local AISHELL-4 excerpt is placed at audio_path."
        ),
    }


def apply_staging_plan_to_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    updated = dict(mapping)
    updated["mapping_status"] = "reference_template_staged"
    updated["staging_status"] = "awaiting_local_audio_download"
    updated["scaffold_note"] = (
        "Reference template staged under resources/external_sanity_check/aishell4. "
        "Audio excerpt must be downloaded locally from OpenSLR; no audio is bundled in this repo."
    )
    return updated


def build_plan_row(mapping: dict[str, Any], audio_path: Path, reference_path: Path) -> dict[str, str]:
    return {
        "dataset_name": str(mapping.get("dataset_name", "AISHELL-4")),
        "slice_id": str(mapping.get("slice_id", "")),
        "label": str(mapping.get("label", "external/sanity-check")),
        "license_status": str(mapping.get("license_status", "")),
        "audio_path": str(mapping.get("audio_path", "")),
        "reference_path": str(mapping.get("reference_path", "")),
        "audio_staged": str(audio_path.exists()),
        "reference_template_staged": str(reference_path.exists()),
        "staging_status": str(mapping.get("staging_status", "awaiting_local_audio_download")),
        "download_source": AISHELL4_DOWNLOAD_URL,
        "result_label": "external/sanity-check",
        "staging_note": (
            "Staging plan only: create paths and reference template; "
            "operator must download one short meeting excerpt before audio eval."
        ),
    }


def build_summary_rows(plan_row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"metric": "staging_status", "value": plan_row["staging_status"], "label": "external/sanity-check"},
        {"metric": "reference_template_staged", "value": plan_row["reference_template_staged"], "label": "external/sanity-check"},
        {"metric": "audio_staged", "value": plan_row["audio_staged"], "label": "external/sanity-check"},
    ]


def build_summary_lines(plan_row: dict[str, str], summary_rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation Audio Excerpt Staging Plan (external/sanity-check)",
        "",
        "Label: `external/sanity-check` — prepares local paths and a reference template for one "
        "AISHELL-4 excerpt. Does not download audio or claim benchmark execution.",
        "",
        "| metric | value | label |",
        "| --- | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(f"| {row['metric']} | {row['value']} | {row['label']} |")
    lines.extend(
        [
            "",
            "## Operator steps",
            "",
            f"1. Confirm license remains research-only (`{plan_row['license_status']}`).",
            f"2. Download one short AISHELL-4 meeting excerpt from {plan_row['download_source']}.",
            f"3. Place audio at `{plan_row['audio_path']}`.",
            f"4. Fill segments in `{plan_row['reference_path']}`.",
            f"5. Re-run `python3 -m src.external_validation_mini_sanity_check`.",
            "",
            f"- Note: {plan_row['staging_note']}",
        ]
    )
    return lines


def write_mapping_artifacts(mapping: dict[str, Any]) -> tuple[Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / "external_validation_slice_mapping.json"
    csv_path = tables_dir / "external_validation_slice_mapping.csv"
    json_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerow(build_mapping_row(mapping))
    return csv_path, json_path


def write_outputs(
    plan_row: dict[str, str],
    summary_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path]:
    table_dir = PROJECT_ROOT / "results" / "tables"
    figure_dir = PROJECT_ROOT / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / "external_validation_audio_excerpt_staging_plan.csv"
    json_path = table_dir / "external_validation_audio_excerpt_staging_plan.json"
    summary_csv_path = table_dir / "external_validation_audio_excerpt_staging_plan_summary.csv"
    summary_json_path = table_dir / "external_validation_audio_excerpt_staging_plan_summary.json"
    md_path = figure_dir / "external_validation_audio_excerpt_staging_plan.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        writer.writerow(plan_row)
    json_path.write_text(json.dumps(plan_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(build_summary_lines(plan_row, summary_rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, summary_csv_path, summary_json_path, md_path


def stage_audio_excerpt_plan() -> dict[str, str]:
    mapping = load_slice_mapping()
    if not mapping:
        raise FileNotFoundError("external_validation_slice_mapping.json is missing")

    audio_path = PROJECT_ROOT / str(mapping.get("audio_path", ""))
    reference_path = PROJECT_ROOT / str(mapping.get("reference_path", ""))
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    reference_template = build_reference_template(mapping)
    reference_path.write_text(json.dumps(reference_template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    updated_mapping = apply_staging_plan_to_mapping(mapping)
    write_mapping_artifacts(updated_mapping)

    plan_row = build_plan_row(updated_mapping, audio_path, reference_path)
    summary_rows = build_summary_rows(plan_row)
    write_outputs(plan_row, summary_rows)
    return plan_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AISHELL-4 audio excerpt staging paths and reference template."
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    _ = load_config()
    plan_row = stage_audio_excerpt_plan()
    print(f"Staging status: {plan_row['staging_status']}")
    print(f"Reference template staged: {plan_row['reference_template_staged']}")
    print(f"Audio staged: {plan_row['audio_staged']}")


if __name__ == "__main__":
    main()
