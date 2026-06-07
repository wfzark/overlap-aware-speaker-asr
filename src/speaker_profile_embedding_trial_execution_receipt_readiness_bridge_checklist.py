from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "case_id",
    "readiness_status",
    "swapped_bias_detected",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_readiness_row() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_receipt_readiness.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(readiness: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(readiness.get("case_id", "NoOverlap"))
    readiness_status = str(readiness.get("readiness_status", "receipt_not_ready"))
    swapped_bias = str(readiness.get("swapped_bias_detected", ""))
    return [
        {
            "checklist_order": "1",
            "case_id": case_id,
            "readiness_status": readiness_status,
            "swapped_bias_detected": swapped_bias,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_execution_receipt_readiness.md",
            "receipt_target": "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
            "checklist_goal": f"Verify embedding receipt readiness for {case_id} before filling the execution receipt.",
            "bridge_note": (
                f"Readiness reports readiness_status={readiness_status} with swapped_bias_detected={swapped_bias}; "
                "confirm receipt readiness before any voiceprint run."
            ),
            "next_gate": "Confirm this bridge before claiming any voiceprint or embedding attribution success.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Trial Execution Receipt Readiness Bridge Checklist",
        "",
        "This generated checklist turns the embedding receipt readiness rollup into a bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim voiceprint success.",
        "",
        "| checklist_order | case_id | readiness_status | swapped_bias_detected | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['case_id']} | {row['readiness_status']} | "
            f"{row['swapped_bias_detected']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_readiness_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_readiness_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_readiness_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_readiness_row())
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial execution receipt readiness bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt readiness bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt readiness bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
