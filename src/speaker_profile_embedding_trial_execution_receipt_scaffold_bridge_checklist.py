from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "case_id",
    "scaffold_status",
    "preflight_pass",
    "swapped_bias_detected",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_receipt_scaffold() -> dict[str, str]:
    scaffold_path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_receipt_scaffold.json"
    if not scaffold_path.exists():
        return {}
    payload = json.loads(scaffold_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(scaffold_row: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(scaffold_row.get("case_id", "NoOverlap"))
    scaffold_status = str(scaffold_row.get("scaffold_status", "receipt_scaffold_only"))
    preflight_pass = str(scaffold_row.get("preflight_pass", ""))
    swapped_bias = str(scaffold_row.get("swapped_bias_detected", ""))
    return [
        {
            "checklist_order": "1",
            "case_id": case_id,
            "scaffold_status": scaffold_status,
            "preflight_pass": preflight_pass,
            "swapped_bias_detected": swapped_bias,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_execution_receipt_scaffold.md",
            "receipt_target": "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
            "checklist_goal": (
                f"Verify the embedding execution receipt scaffold for {case_id} before any voiceprint run."
            ),
            "bridge_note": (
                f"Receipt scaffold remains {scaffold_status} with preflight_pass={preflight_pass} "
                f"and swapped_bias_detected={swapped_bias}; confirm scaffold context before filling the execution receipt."
            ),
            "next_gate": "Confirm this bridge before claiming any voiceprint or embedding model execution.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Trial Execution Receipt Scaffold Bridge Checklist",
        "",
        "This generated checklist turns the embedding execution receipt scaffold into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim voiceprint success.",
        "",
        "| checklist_order | case_id | scaffold_status | preflight_pass | swapped_bias_detected | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['case_id']} | {row['scaffold_status']} | {row['preflight_pass']} | "
            f"{row['swapped_bias_detected']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_scaffold_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_scaffold_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_scaffold_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    scaffold_row = load_receipt_scaffold()
    rows = build_bridge_checklist_rows(scaffold_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial execution receipt scaffold bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt scaffold bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt scaffold bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
