from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "handoff_status",
    "overall_state",
    "case_scope",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_handoff() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_go_no_go_board_handoff.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(handoff: dict[str, str]) -> list[dict[str, str]]:
    if not handoff:
        return []
    handoff_status = str(handoff.get("handoff_status", "speaker_profile_go_handoff_pending"))
    overall_state = str(handoff.get("overall_state", "execution_not_ready"))
    case_scope = str(handoff.get("case_scope", "NoOverlap"))
    return [
        {
            "checklist_order": "1",
            "handoff_status": handoff_status,
            "overall_state": overall_state,
            "case_scope": case_scope,
            "prerequisite_artifact": "results/figures/speaker_profile_go_no_go_board_handoff.md",
            "receipt_target": "results/figures/speaker_profile_embedding_trial_execution_scaffold_readiness.md",
            "checklist_goal": (
                "Verify the speaker-profile go-no-go handoff before reopening the embedding trial execution scaffold."
            ),
            "bridge_note": (
                f"Go-no-go handoff reports handoff_status={handoff_status} with overall_state="
                f"{overall_state} for {case_scope}; attribution claims remain blocked."
            ),
            "next_gate": (
                "Confirm this bridge before opening the speaker profile embedding trial execution scaffold readiness."
            ),
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Go-No-Go Board Handoff Bridge Checklist",
        "",
        "This generated checklist connects the speaker-profile go-no-go handoff to the embedding trial execution scaffold. "
        "It does not claim speaker identification success.",
        "",
        "| checklist_order | handoff_status | overall_state | case_scope | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['handoff_status']} | {row['overall_state']} | "
            f"{row['case_scope']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_go_no_go_board_handoff_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_go_no_go_board_handoff_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_go_no_go_board_handoff_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_handoff())
    if not rows:
        print("Speaker profile go-no-go handoff not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote speaker profile go-no-go handoff bridge checklist CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go handoff bridge checklist JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go handoff bridge checklist note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
