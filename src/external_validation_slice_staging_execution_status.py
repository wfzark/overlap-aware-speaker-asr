from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


STATUS_COLUMNS = [
    "scope",
    "dataset_name",
    "handoff_status",
    "blocker",
    "receipt_scaffold_status",
    "execution_receipt_status",
    "execution_chain_status",
    "status_note",
]


def load_json_dict(path_rel: str) -> dict[str, str]:
    path = PROJECT_ROOT / path_rel
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_receipt_template_status(path_rel: str) -> str:
    path = PROJECT_ROOT / path_rel
    if not path.exists():
        return "missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return str(first.get("execution_status", "unknown"))
    return "unknown"


def build_status_row(
    handoff: dict[str, str],
    receipt_scaffold: dict[str, str],
    execution_receipt_status: str,
) -> dict[str, str]:
    dataset_name = str(handoff.get("dataset_name", receipt_scaffold.get("dataset_name", "AISHELL-4")))
    handoff_status = str(handoff.get("handoff_status", "staging_handoff_ready"))
    blocker = str(handoff.get("blocker", receipt_scaffold.get("blocker", "license_confirmation_pending")))
    scaffold_status = str(receipt_scaffold.get("scaffold_status", "missing"))
    chain_ready = scaffold_status == "receipt_scaffold_only" and blocker != ""
    chain_status = "execution_chain_ready" if chain_ready else "execution_chain_in_progress"
    return {
        "scope": "external_validation_slice_staging_execution_chain",
        "dataset_name": dataset_name,
        "handoff_status": handoff_status,
        "blocker": blocker,
        "receipt_scaffold_status": scaffold_status,
        "execution_receipt_status": execution_receipt_status,
        "execution_chain_status": chain_status,
        "status_note": (
            "external/sanity-check staging execution-chain rollup for one narrow slice; "
            "external audio staging and benchmark evaluation are not claimed."
        ),
    }


def build_status_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# External Validation Slice Staging Execution Status",
        "",
        "This generated note rolls up the external slice staging execution-chain status. "
        "It does not claim external audio download or benchmark execution.",
        "",
        "| scope | dataset_name | handoff_status | blocker | receipt_scaffold_status | execution_receipt_status | execution_chain_status | status_note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['scope']} | {row['dataset_name']} | {row['handoff_status']} | {row['blocker']} | "
            f"{row['receipt_scaffold_status']} | {row['execution_receipt_status']} | {row['execution_chain_status']} | "
            f"{row['status_note']} |"
        ),
    ]
    return lines


def write_outputs(status_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "external_validation_slice_staging_execution_status.csv"
    json_path = tables_dir / "external_validation_slice_staging_execution_status.json"
    md_path = figures_dir / "external_validation_slice_staging_execution_status.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerow(status_row)
    json_path.write_text(json.dumps(status_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_status_lines(status_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    handoff = load_json_dict("results/tables/external_validation_slice_staging_readiness_handoff.json")
    receipt_scaffold = load_json_dict("results/tables/external_validation_slice_staging_handoff_receipt_scaffold.json")
    execution_receipt_status = load_receipt_template_status(
        "results/tables/external_validation_slice_staging_handoff_receipt.json"
    )
    status_row = build_status_row(handoff, receipt_scaffold, execution_receipt_status)
    csv_path, json_path, md_path = write_outputs(status_row)
    print(f"Wrote external validation slice staging execution status CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation slice staging execution status JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation slice staging execution status note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Execution chain status: {status_row['execution_chain_status']}")


if __name__ == "__main__":
    main()
