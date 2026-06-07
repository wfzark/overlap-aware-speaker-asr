from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


SCAFFOLD_COLUMNS = [
    "dataset_name",
    "label",
    "license_status",
    "confirmation_status",
    "confirmation_step",
    "expected_writeback",
    "scaffold_note",
]

RECEIPT_COLUMNS = [
    "execution_status",
    "confirmation_scope",
    "dataset_name",
    "writeback_note",
]


def load_license_gate_row() -> dict[str, str]:
    gate_path = PROJECT_ROOT / "results" / "tables" / "external_validation_license_gate.csv"
    if not gate_path.exists():
        return {}
    with gate_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return {}


def build_scaffold_row(gate_row: dict[str, str]) -> dict[str, str]:
    dataset_name = str(gate_row.get("dataset_name", "AISHELL-4"))
    license_status = str(gate_row.get("license_status", "pending_confirmation"))
    return {
        "dataset_name": dataset_name,
        "label": str(gate_row.get("label", "external/sanity-check")),
        "license_status": license_status,
        "confirmation_status": "template_only",
        "confirmation_step": (
            f"Record the official {dataset_name} license decision in the slice receipt before any audio staging begins."
        ),
        "expected_writeback": "results/tables/external_validation_slice_receipt.json",
        "scaffold_note": (
            "License confirmation scaffold only; no license decision has been recorded and no external audio has been staged."
        ),
    }


def build_scaffold_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# External Validation License Confirmation Scaffold",
        "",
        "This generated scaffold records the writeback slot for a future license confirmation decision. "
        "It does not claim that any license has been confirmed or that external audio has been staged.",
        "",
        "| dataset_name | label | license_status | confirmation_status | confirmation_step | expected_writeback | scaffold_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['dataset_name']} | {row['label']} | {row['license_status']} | {row['confirmation_status']} | "
            f"{row['confirmation_step']} | {row['expected_writeback']} | {row['scaffold_note']} |"
        ),
    ]
    return lines


def build_scaffold_receipt_rows(scaffold_row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "execution_status": "scaffold_documented",
            "confirmation_scope": "single_dataset_license_decision",
            "dataset_name": str(scaffold_row.get("dataset_name", "")),
            "writeback_note": (
                "License confirmation scaffold documented; external audio staging remains blocked until a decision is recorded."
            ),
        }
    ]


def build_scaffold_receipt_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation License Confirmation Scaffold Receipt",
        "",
        "This receipt records the license confirmation scaffold writeback. It does not claim benchmark execution.",
        "",
        "| execution_status | confirmation_scope | dataset_name | writeback_note |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['execution_status']} | {row['confirmation_scope']} | {row['dataset_name']} | "
            f"{row['writeback_note']} |"
        )
    return lines


def write_outputs(
    scaffold_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "external_validation_license_confirmation_scaffold.csv"
    json_path = tables_dir / "external_validation_license_confirmation_scaffold.json"
    md_path = figures_dir / "external_validation_license_confirmation_scaffold.md"
    receipt_json_path = tables_dir / "external_validation_license_confirmation_scaffold_receipt.json"
    receipt_md_path = figures_dir / "external_validation_license_confirmation_scaffold_receipt.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SCAFFOLD_COLUMNS)
        writer.writeheader()
        writer.writerow(scaffold_row)
    json_path.write_text(json.dumps(scaffold_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_scaffold_lines(scaffold_row)) + "\n", encoding="utf-8")
    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_scaffold_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path, receipt_json_path, receipt_md_path


def main() -> None:
    gate_row = load_license_gate_row()
    scaffold_row = build_scaffold_row(gate_row)
    receipt_rows = build_scaffold_receipt_rows(scaffold_row)
    csv_path, json_path, md_path, receipt_json_path, receipt_md_path = write_outputs(
        scaffold_row,
        receipt_rows,
    )
    print(
        "Wrote external validation license confirmation scaffold CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation scaffold JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation scaffold note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation scaffold receipt JSON: "
        f"{receipt_json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation scaffold receipt note: "
        f"{receipt_md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
