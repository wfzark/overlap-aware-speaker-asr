from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


HANDOFF_COLUMNS = [
    "handoff_order",
    "frontier_name",
    "readiness_status",
    "recommended_action",
    "expected_inputs",
    "expected_outputs",
    "handoff_note",
]

FRONTIER_RECEIPTS = [
    ("meeteval_compatibility", "meeteval_readiness_status", "meeteval_cpwer_execution_receipt.json"),
    ("speaker_profile", "speaker_profile_readiness_status", "speaker_profile_embedding_trial_execution_receipt.json"),
    (
        "external_validation",
        "external_staging_readiness_status",
        "external_validation_slice_staging_handoff_receipt.json",
    ),
]


def load_status_row() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_status.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_handoff_rows(status_row: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for order, (frontier_name, status_key, receipt_file) in enumerate(FRONTIER_RECEIPTS, start=1):
        readiness = str(status_row.get(status_key, "receipt_not_ready"))
        receipt_path = f"results/tables/{receipt_file}"
        action = (
            f"Update execution_status in {receipt_path} after a real frontier run and bridge verification."
            if readiness == "receipt_ready_to_fill"
            else "Complete remaining receipt readiness checks before opening the execution receipt."
        )
        rows.append(
            {
                "handoff_order": str(order),
                "frontier_name": frontier_name,
                "readiness_status": readiness,
                "recommended_action": action,
                "expected_inputs": "results/figures/frontier_execution_receipt_queue_status.md; per-frontier receipt readiness bridge checklist.",
                "expected_outputs": receipt_path,
                "handoff_note": (
                    f"Receipt-fill handoff for {frontier_name} while readiness_status={readiness}; "
                    "no benchmark execution or external audio staging is claimed."
                ),
            }
        )
    return rows


def build_handoff_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Queue Handoff",
        "",
        "This generated note turns the unified receipt readiness rollup into per-frontier receipt-fill actions. "
        "It remains experimental/frontier coordination only and does not claim benchmark completion.",
        "",
        "| handoff_order | frontier_name | readiness_status | recommended_action | expected_inputs | expected_outputs | handoff_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['handoff_order']} | {row['frontier_name']} | {row['readiness_status']} | "
            f"{row['recommended_action']} | {row['expected_inputs']} | {row['expected_outputs']} | {row['handoff_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_handoff.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_handoff.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_handoff.md"

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
    print(f"Wrote frontier execution receipt queue handoff CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue handoff JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue handoff note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
