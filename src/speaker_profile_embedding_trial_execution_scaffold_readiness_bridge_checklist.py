from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "readiness_status",
    "case_id",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_readiness() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_scaffold_readiness.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(readiness: dict[str, str]) -> list[dict[str, str]]:
    if not readiness:
        return []
    readiness_status = str(readiness.get("readiness_status", "scaffold_not_ready"))
    case_id = str(readiness.get("case_id", "NoOverlap"))
    return [
        {
            "checklist_order": "1",
            "readiness_status": readiness_status,
            "case_id": case_id,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_execution_scaffold_readiness.md",
            "receipt_target": "results/figures/speaker_profile_embedding_trial_execution_preflight.md",
            "checklist_goal": (
                f"Verify execution scaffold readiness for {case_id} before opening the execution preflight."
            ),
            "bridge_note": (
                f"Scaffold readiness reports readiness_status={readiness_status} for case_id={case_id}; "
                "confirm handoff completion before advancing to execution preflight."
            ),
            "next_gate": "Confirm this bridge before opening the embedding trial execution preflight.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Trial Execution Scaffold Readiness Bridge Checklist",
        "",
        "This generated checklist connects execution scaffold readiness to the execution preflight. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        "| checklist_order | readiness_status | case_id | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['readiness_status']} | {row['case_id']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_scaffold_readiness_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_scaffold_readiness_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_scaffold_readiness_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_readiness())
    if not rows:
        print("Execution scaffold readiness not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial execution scaffold readiness bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution scaffold readiness bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution scaffold readiness bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
