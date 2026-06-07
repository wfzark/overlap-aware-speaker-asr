from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_COLUMNS = [
    "bridge_order",
    "dataset_name",
    "confirmation_status",
    "license_status",
    "prerequisite_artifact",
    "receipt_target",
    "bridge_goal",
    "bridge_note",
    "next_gate",
]


def load_confirmation_scaffold() -> dict[str, str]:
    scaffold_path = PROJECT_ROOT / "results" / "tables" / "external_validation_license_confirmation_scaffold.json"
    if not scaffold_path.exists():
        return {}
    payload = json.loads(scaffold_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_rows(scaffold: dict[str, str]) -> list[dict[str, str]]:
    dataset_name = str(scaffold.get("dataset_name", "AISHELL-4"))
    confirmation_status = str(scaffold.get("confirmation_status", "template_only"))
    license_status = str(scaffold.get("license_status", "pending_confirmation"))
    return [
        {
            "bridge_order": "1",
            "dataset_name": dataset_name,
            "confirmation_status": confirmation_status,
            "license_status": license_status,
            "prerequisite_artifact": "results/figures/external_validation_license_confirmation_scaffold_bridge_checklist.md",
            "receipt_target": "results/figures/external_validation_slice_receipt.md",
            "bridge_goal": (
                f"Connect the license confirmation scaffold bridge to the slice receipt for {dataset_name}."
            ),
            "bridge_note": (
                f"Confirmation remains {confirmation_status} with license_status={license_status}; "
                "no external audio staging or benchmark execution is claimed."
            ),
            "next_gate": "Confirm this bridge before opening the external slice receipt target.",
        }
    ]


def build_bridge_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation License Confirmation Receipt Bridge",
        "",
        "This generated bridge links the license confirmation scaffold bridge checklist to the slice receipt. "
        "It remains external/sanity-check coordination only and does not claim benchmark execution.",
        "",
        "| bridge_order | dataset_name | confirmation_status | license_status | prerequisite_artifact | receipt_target | bridge_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['bridge_order']} | {row['dataset_name']} | {row['confirmation_status']} | "
            f"{row['license_status']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['bridge_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "external_validation_license_confirmation_receipt_bridge.csv"
    json_path = tables_dir / "external_validation_license_confirmation_receipt_bridge.json"
    md_path = figures_dir / "external_validation_license_confirmation_receipt_bridge.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    scaffold = load_confirmation_scaffold()
    rows = build_bridge_rows(scaffold)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote external validation license confirmation receipt bridge CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation receipt bridge JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote external validation license confirmation receipt bridge note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
