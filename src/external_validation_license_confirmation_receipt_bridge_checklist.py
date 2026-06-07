from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "dataset_name",
    "confirmation_status",
    "license_status",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_receipt_bridge_row() -> dict[str, str]:
    bridge_path = PROJECT_ROOT / "results" / "tables" / "external_validation_license_confirmation_receipt_bridge.json"
    if not bridge_path.exists():
        return {}
    payload = json.loads(bridge_path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[0]
    return {}


def build_bridge_checklist_rows(bridge_row: dict[str, str]) -> list[dict[str, str]]:
    dataset_name = str(bridge_row.get("dataset_name", "AISHELL-4"))
    confirmation_status = str(bridge_row.get("confirmation_status", "template_only"))
    license_status = str(bridge_row.get("license_status", "pending_confirmation"))
    return [
        {
            "checklist_order": "1",
            "dataset_name": dataset_name,
            "confirmation_status": confirmation_status,
            "license_status": license_status,
            "prerequisite_artifact": "results/figures/external_validation_license_confirmation_receipt_bridge.md",
            "receipt_target": "results/figures/external_validation_slice_receipt.md",
            "checklist_goal": (
                f"Verify the license confirmation receipt bridge for {dataset_name} before opening the slice receipt."
            ),
            "bridge_note": (
                f"Receipt bridge remains confirmation_status={confirmation_status} with license_status={license_status}; "
                "no external audio staging or benchmark execution is claimed."
            ),
            "next_gate": "Confirm this bridge before opening the external slice receipt target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation License Confirmation Receipt Bridge Checklist",
        "",
        "This generated checklist turns the license confirmation receipt bridge into a row-by-row verification path. "
        "It remains external/sanity-check coordination only and does not claim benchmark execution.",
        "",
        "| checklist_order | dataset_name | confirmation_status | license_status | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['dataset_name']} | {row['confirmation_status']} | "
            f"{row['license_status']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "external_validation_license_confirmation_receipt_bridge_checklist.csv"
    json_path = tables_dir / "external_validation_license_confirmation_receipt_bridge_checklist.json"
    md_path = figures_dir / "external_validation_license_confirmation_receipt_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    bridge_row = load_receipt_bridge_row()
    rows = build_bridge_checklist_rows(bridge_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote external validation license confirmation receipt bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation receipt bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation receipt bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
