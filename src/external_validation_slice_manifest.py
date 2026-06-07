from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


MANIFEST_COLUMNS = [
    "dataset_name",
    "slice_id",
    "label",
    "license_status",
    "mapping_status",
    "audio_path",
    "reference_path",
    "staging_status",
    "manifest_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "slice_scope",
    "dataset_name",
    "staging_status",
    "writeback_note",
]


def load_slice_mapping() -> dict[str, Any]:
    mapping_path = PROJECT_ROOT / "results" / "tables" / "external_validation_slice_mapping.json"
    if not mapping_path.exists():
        return {}
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def build_manifest_row(mapping: dict[str, Any]) -> dict[str, str]:
    license_status = str(mapping.get("license_status", "pending_confirmation"))
    staging_status = "blocked_by_license_gate" if license_status == "pending_confirmation" else "ready_for_staging"
    return {
        "dataset_name": str(mapping.get("dataset_name", "AISHELL-4")),
        "slice_id": str(mapping.get("slice_id", "")),
        "label": str(mapping.get("label", "external/sanity-check")),
        "license_status": license_status,
        "mapping_status": str(mapping.get("mapping_status", "scaffold_only")),
        "audio_path": str(mapping.get("audio_path", "")),
        "reference_path": str(mapping.get("reference_path", "")),
        "staging_status": staging_status,
        "manifest_note": (
            "Manifest-only staging plan for the first external slice. "
            "No external audio has been downloaded or evaluated."
        ),
    }


def build_manifest_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# External Validation Slice Manifest",
        "",
        "This generated manifest records the staging plan for the first external sanity-check slice. "
        "It does not claim that any external benchmark has been executed.",
        "",
        "| dataset_name | slice_id | label | license_status | mapping_status | audio_path | reference_path | staging_status | manifest_note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['dataset_name']} | {row['slice_id']} | {row['label']} | {row['license_status']} | "
            f"{row['mapping_status']} | {row['audio_path']} | {row['reference_path']} | {row['staging_status']} | "
            f"{row['manifest_note']} |"
        ),
    ]
    return lines


def build_manifest_receipt_rows(manifest_row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "execution_status": "manifest_complete",
            "slice_scope": "single_short_meeting_excerpt",
            "dataset_name": str(manifest_row.get("dataset_name", "")),
            "staging_status": str(manifest_row.get("staging_status", "")),
            "writeback_note": "Slice manifest documented; external audio staging remains blocked until license confirmation is recorded.",
        }
    ]


def build_manifest_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation Slice Manifest Receipt",
        "",
        "This receipt records the slice-manifest writeback. It does not claim external benchmark execution.",
        "",
        "| execution_status | slice_scope | dataset_name | staging_status | writeback_note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['slice_scope']} | {row['dataset_name']} | "
            f"{row['staging_status']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(
    manifest_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv_path = tables_dir / "external_validation_slice_manifest.csv"
    manifest_json_path = tables_dir / "external_validation_slice_manifest.json"
    manifest_md_path = figures_dir / "external_validation_slice_manifest.md"
    receipt_json_path = tables_dir / "external_validation_slice_manifest_receipt.json"
    receipt_md_path = figures_dir / "external_validation_slice_manifest_receipt.md"

    with manifest_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(manifest_row)
    manifest_json_path.write_text(json.dumps(manifest_row, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_md_path.write_text("\n".join(build_manifest_lines(manifest_row)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_manifest_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return manifest_csv_path, manifest_json_path, manifest_md_path, receipt_json_path, receipt_md_path


def main() -> None:
    mapping = load_slice_mapping()
    manifest_row = build_manifest_row(mapping)
    receipt_rows = build_manifest_receipt_rows(manifest_row)
    manifest_csv_path, manifest_json_path, manifest_md_path, receipt_json_path, receipt_md_path = write_outputs(
        manifest_row,
        receipt_rows,
    )
    print(f"Wrote external slice manifest CSV: {manifest_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external slice manifest JSON: {manifest_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external slice manifest note: {manifest_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external slice manifest receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external slice manifest receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
