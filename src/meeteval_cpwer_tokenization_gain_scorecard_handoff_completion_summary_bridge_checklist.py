from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "queue_status",
    "handoff_status",
    "adapted_and_aligned_count",
    "case_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_completion_summary() -> dict[str, str]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary.json"
    )
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    if not summary:
        return []
    queue_status = str(summary.get("queue_status", "queue_in_progress"))
    handoff_status = str(summary.get("handoff_status", "tokenization_gain_handoff_pending"))
    adapted_and_aligned_count = str(summary.get("adapted_and_aligned_count", "0"))
    case_count = str(summary.get("case_count", "0"))
    return [
        {
            "checklist_order": "1",
            "queue_status": queue_status,
            "handoff_status": handoff_status,
            "adapted_and_aligned_count": adapted_and_aligned_count,
            "case_count": case_count,
            "prerequisite_artifact": (
                "results/figures/meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary.md"
            ),
            "receipt_target": "results/figures/meeteval_tokenization_adaptation_handoff.md",
            "checklist_goal": (
                "Verify tokenization gain handoff completion before advancing the tokenization adaptation handoff."
            ),
            "bridge_note": (
                f"Gain handoff completion reports queue_status={queue_status} with handoff_status="
                f"{handoff_status} at {adapted_and_aligned_count}/{case_count} adapted-and-aligned cases."
            ),
            "next_gate": "Confirm this bridge before opening the tokenization adaptation handoff.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Tokenization Gain Scorecard Handoff Completion Summary Bridge Checklist",
        "",
        "This generated checklist connects tokenization gain handoff completion to the tokenization adaptation handoff. "
        "It does not claim full MeetEval benchmark completion.",
        "",
        "| checklist_order | queue_status | handoff_status | adapted_and_aligned_count | case_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['queue_status']} | {row['handoff_status']} | "
            f"{row['adapted_and_aligned_count']} | {row['case_count']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        tables_dir
        / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary_bridge_checklist.csv"
    )
    json_path = (
        tables_dir
        / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary_bridge_checklist.json"
    )
    md_path = (
        figures_dir
        / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary_bridge_checklist.md"
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_completion_summary())
    if not rows:
        print("Tokenization gain handoff completion summary not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER tokenization gain handoff completion summary bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER tokenization gain handoff completion summary bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER tokenization gain handoff completion summary bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
