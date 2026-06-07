from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "case_id",
    "execution_chain_status",
    "swapped_bias_detected",
    "preflight_pass",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_status_row() -> dict[str, str]:
    status_path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_status.json"
    if not status_path.exists():
        return {}
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(status_row: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(status_row.get("case_id", "NoOverlap"))
    chain_status = str(status_row.get("execution_chain_status", "execution_chain_in_progress"))
    preflight_pass = str(status_row.get("preflight_pass", ""))
    swapped_bias = str(status_row.get("swapped_bias_detected", ""))
    return [
        {
            "checklist_order": "1",
            "case_id": case_id,
            "execution_chain_status": chain_status,
            "swapped_bias_detected": swapped_bias,
            "preflight_pass": preflight_pass,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_execution_status.md",
            "receipt_target": "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
            "checklist_goal": (
                f"Verify the embedding execution-chain status rollup for {case_id} before any voiceprint run."
            ),
            "bridge_note": (
                f"Status rollup reports execution_chain_status={chain_status} with swapped_bias_detected={swapped_bias}; "
                "confirm chain readiness before filling the embedding execution receipt."
            ),
            "next_gate": "Confirm this bridge before claiming any voiceprint or embedding attribution success.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Trial Execution Status Bridge Checklist",
        "",
        "This generated checklist turns the embedding execution-chain status rollup into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim voiceprint success.",
        "",
        "| checklist_order | case_id | execution_chain_status | swapped_bias_detected | preflight_pass | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['case_id']} | {row['execution_chain_status']} | "
            f"{row['swapped_bias_detected']} | {row['preflight_pass']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_status_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_status_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_status_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    status_row = load_status_row()
    rows = build_bridge_checklist_rows(status_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial execution status bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution status bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution status bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
