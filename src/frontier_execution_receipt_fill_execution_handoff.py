from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


HANDOFF_COLUMNS = [
    "handoff_order",
    "frontier_name",
    "fill_execution_status",
    "recommended_action",
    "expected_inputs",
    "expected_outputs",
    "handoff_note",
]

FRONTIER_HANDOFFS = [
    (
        "meeteval_compatibility",
        "meeteval_fill_execution_status",
        "results/tables/meeteval_cpwer_execution_receipt.json",
    ),
    (
        "speaker_profile",
        "speaker_profile_fill_execution_status",
        "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
    ),
    (
        "external_validation",
        "external_staging_fill_execution_status",
        "results/tables/external_validation_slice_staging_handoff_receipt.json",
    ),
]


def load_status_row() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_fill_execution_status.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_handoff_rows(status_row: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for order, (frontier_name, status_key, receipt_path) in enumerate(FRONTIER_HANDOFFS, start=1):
        fill_execution_status = str(status_row.get(status_key, "receipt_missing"))
        if fill_execution_status == "awaiting_fill":
            action = (
                f"Execute the real frontier run, then update execution_status in {receipt_path} "
                "and attach the fill evidence note."
            )
        elif fill_execution_status == "fill_complete":
            action = f"Review the filled receipt at {receipt_path} and archive the fill evidence note."
        else:
            action = "Resolve receipt blockers before attempting fill execution."
        rows.append(
            {
                "handoff_order": str(order),
                "frontier_name": frontier_name,
                "fill_execution_status": fill_execution_status,
                "recommended_action": action,
                "expected_inputs": (
                    "results/figures/frontier_execution_receipt_fill_execution_status.md; "
                    "fill execution packet and handoff bridge checklist."
                ),
                "expected_outputs": receipt_path,
                "handoff_note": (
                    f"Fill execution handoff for {frontier_name} while fill_execution_status={fill_execution_status}; "
                    "no benchmark execution or external audio staging is claimed until receipts are filled."
                ),
            }
        )
    return rows


def build_handoff_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Fill Execution Handoff",
        "",
        "This generated note turns the unified fill execution status rollup into per-frontier fill execution actions. "
        "It remains experimental/frontier coordination only and does not claim benchmark completion.",
        "",
        "| handoff_order | frontier_name | fill_execution_status | recommended_action | expected_inputs | expected_outputs | handoff_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['handoff_order']} | {row['frontier_name']} | {row['fill_execution_status']} | "
            f"{row['recommended_action']} | {row['expected_inputs']} | {row['expected_outputs']} | {row['handoff_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_fill_execution_handoff.csv"
    json_path = tables_dir / "frontier_execution_receipt_fill_execution_handoff.json"
    md_path = figures_dir / "frontier_execution_receipt_fill_execution_handoff.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_handoff_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_handoff_rows(load_status_row())
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote frontier execution receipt fill execution handoff CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt fill execution handoff JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt fill execution handoff note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
