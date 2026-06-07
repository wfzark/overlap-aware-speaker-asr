from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "case_id",
    "preflight_pass",
    "swapped_bias_detected",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_execution_preflight() -> dict[str, str]:
    preflight_path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_preflight.json"
    if not preflight_path.exists():
        return {}
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(preflight_row: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(preflight_row.get("case_id", "NoOverlap"))
    preflight_pass = str(preflight_row.get("preflight_pass", False))
    swapped_bias = str(preflight_row.get("swapped_bias_detected", False))
    return [
        {
            "checklist_order": "1",
            "case_id": case_id,
            "preflight_pass": preflight_pass,
            "swapped_bias_detected": swapped_bias,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_execution_preflight.md",
            "receipt_target": "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
            "checklist_goal": (
                f"Verify the embedding execution preflight for {case_id} before opening the execution receipt."
            ),
            "bridge_note": (
                f"Execution preflight reports preflight_pass={preflight_pass} with swapped_bias_detected={swapped_bias}; "
                "confirm proxy-data readiness before advancing to voiceprint execution."
            ),
            "next_gate": "Confirm this bridge before opening the embedding trial execution receipt target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Trial Execution Preflight Bridge Checklist",
        "",
        "This generated checklist turns the embedding execution preflight into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim voiceprint success.",
        "",
        "| checklist_order | case_id | preflight_pass | swapped_bias_detected | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['case_id']} | {row['preflight_pass']} | {row['swapped_bias_detected']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_preflight_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_preflight_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_preflight_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    preflight_row = load_execution_preflight()
    rows = build_bridge_checklist_rows(preflight_row)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial execution preflight bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution preflight bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution preflight bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
