from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "operator_case",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_bridge_row() -> dict[str, str]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge.json"
    )
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(bridge_row: dict[str, str]) -> list[dict[str, str]]:
    if not bridge_row:
        return []
    operator_case = str(bridge_row.get("operator_case", "NoOverlap"))
    receipt_target = str(bridge_row.get("receipt_target", ""))
    return [
        {
            "checklist_order": "1",
            "operator_case": operator_case,
            "prerequisite_artifact": str(bridge_row.get("prerequisite_artifact", "")),
            "receipt_target": receipt_target,
            "checklist_goal": (
                f"Verify the speaker-profile operator brief bridge for {operator_case} before reopening the readiness target."
            ),
            "bridge_note": str(bridge_row.get("bridge_note", "")),
            "next_gate": f"Confirm this bridge before opening {receipt_target}.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    return [
        "# Speaker Profile Embedding Trial Execution Receipt Operator Brief Bridge Checklist",
        "",
        "This generated checklist turns the speaker-profile operator brief bridge into an ordered verification path. "
        "It remains experimental/frontier coordination only and does not fill receipts or claim voiceprint success.",
        "",
        "| checklist_order | operator_case | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {row['checklist_order']} | {row['operator_case']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
            for row in rows
        ],
    ]


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_bridge_row())
    if not rows:
        print("Speaker-profile receipt operator brief bridge not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial execution receipt operator brief bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt operator brief bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt operator brief bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
