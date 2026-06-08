from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "readiness_status",
    "trial_case_target",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_readiness() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_handoff_readiness.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(readiness: dict[str, str]) -> list[dict[str, str]]:
    if not readiness:
        return []
    readiness_status = str(readiness.get("readiness_status", "handoff_not_ready"))
    trial_case = str(readiness.get("trial_case_target", "NoOverlap"))
    return [
        {
            "checklist_order": "1",
            "readiness_status": readiness_status,
            "trial_case_target": trial_case,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_handoff_readiness.md",
            "receipt_target": "results/figures/speaker_profile_embedding_trial.md",
            "checklist_goal": (
                "Verify embedding trial handoff readiness before opening the single-case embedding trial scaffold."
            ),
            "bridge_note": (
                f"Handoff readiness reports readiness_status={readiness_status} for trial_case_target={trial_case}; "
                "confirm text-proxy diagnostic completion before advancing to embedding trial."
            ),
            "next_gate": "Confirm this bridge before opening the embedding trial target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Trial Handoff Readiness Bridge Checklist",
        "",
        "This generated checklist connects handoff readiness to the embedding trial scaffold. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        "| checklist_order | readiness_status | trial_case_target | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['readiness_status']} | {row['trial_case_target']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_handoff_readiness_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_handoff_readiness_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_handoff_readiness_bridge_checklist.md"

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
        print("Embedding trial handoff readiness not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding trial handoff readiness bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial handoff readiness bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial handoff readiness bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
