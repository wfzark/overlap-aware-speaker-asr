from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "dataset_name",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_manifest_row() -> dict[str, str]:
    manifest_path = PROJECT_ROOT / "results" / "tables" / "external_validation_slice_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(manifest_row: dict[str, str]) -> list[dict[str, str]]:
    dataset_name = str(manifest_row.get("dataset_name", "AISHELL-4"))
    staging_status = str(manifest_row.get("staging_status", "blocked_by_license_gate"))
    return [
        {
            "checklist_order": "1",
            "dataset_name": dataset_name,
            "prerequisite_artifact": "results/figures/external_validation_slice_manifest.md",
            "receipt_target": "results/figures/external_validation_slice_manifest_receipt.md",
            "checklist_goal": (
                f"Verify the external slice manifest bridge for {dataset_name} before any staging writeback is advanced."
            ),
            "bridge_note": (
                f"Open the slice manifest first, then write back through the manifest receipt while staging remains "
                f"{staging_status}."
            ),
            "next_gate": "Confirm this bridge before opening the slice manifest receipt target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation Slice Manifest Bridge Checklist",
        "",
        "This generated checklist turns the slice manifest into a row-by-row bridge verification path. "
        "It remains external/sanity-check coordination only and does not claim benchmark execution.",
        "",
        "| checklist_order | dataset_name | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['dataset_name']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "external_validation_slice_manifest_bridge_checklist.csv"
    json_path = tables_dir / "external_validation_slice_manifest_bridge_checklist.json"
    md_path = figures_dir / "external_validation_slice_manifest_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    manifest_row = load_manifest_row()
    rows = build_bridge_checklist_rows(manifest_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote external validation slice manifest bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation slice manifest bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation slice manifest bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
