from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "overall_state",
    "go_count",
    "checkpoint_count",
    "case_scope",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_go_no_go_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    if not summary:
        return []
    overall_state = str(summary.get("overall_state", "execution_not_ready"))
    go_count = str(summary.get("go_count", "0"))
    checkpoint_count = str(summary.get("checkpoint_count", "0"))
    case_scope = str(summary.get("case_scope", "NoOverlap"))
    return [
        {
            "checklist_order": "1",
            "overall_state": overall_state,
            "go_count": go_count,
            "checkpoint_count": checkpoint_count,
            "case_scope": case_scope,
            "prerequisite_artifact": "results/figures/speaker_profile_go_no_go_summary.md",
            "receipt_target": "results/figures/speaker_profile_embedding_trial_execution_preflight_readiness.md",
            "checklist_goal": (
                "Verify the speaker-profile go-no-go board before reopening the embedding trial execution preflight."
            ),
            "bridge_note": (
                f"Go-no-go summary reports overall_state={overall_state} with {go_count}/{checkpoint_count} "
                f"go checkpoints for {case_scope}; attribution claims remain blocked."
            ),
            "next_gate": (
                "Confirm this bridge before opening the speaker profile embedding trial execution preflight."
            ),
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Go-No-Go Board Bridge Checklist",
        "",
        "This generated checklist connects the speaker-profile go-no-go board to the embedding trial execution preflight. "
        "It does not claim speaker identification success.",
        "",
        "| checklist_order | overall_state | go_count | checkpoint_count | case_scope | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['overall_state']} | {row['go_count']} | "
            f"{row['checkpoint_count']} | {row['case_scope']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_go_no_go_board_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_go_no_go_board_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_go_no_go_board_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_summary())
    if not rows:
        print("Speaker profile go-no-go summary not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote speaker profile go-no-go board bridge checklist CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go board bridge checklist JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go board bridge checklist note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
