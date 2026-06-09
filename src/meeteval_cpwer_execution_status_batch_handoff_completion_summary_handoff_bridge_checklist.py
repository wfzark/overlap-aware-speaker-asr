from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "handoff_status",
    "queue_status",
    "complete_handoff_count",
    "total_handoff_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_handoff_rows() -> list[dict[str, str]]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff.json"
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(handoff_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not handoff_rows:
        return []
    handoff = handoff_rows[0]
    handoff_status = str(handoff.get("handoff_status", "batch_handoff_completion_handoff_pending"))
    queue_status = str(handoff.get("queue_status", "queue_in_progress"))
    complete_handoff_count = str(handoff.get("complete_handoff_count", "0"))
    total_handoff_count = str(handoff.get("total_handoff_count", "0"))
    return [
        {
            "checklist_order": "1",
            "handoff_status": handoff_status,
            "queue_status": queue_status,
            "complete_handoff_count": complete_handoff_count,
            "total_handoff_count": total_handoff_count,
            "prerequisite_artifact": (
                "results/figures/meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff.md"
            ),
            "receipt_target": "results/figures/meeteval_cpwer_official_execution_completion_summary.md",
            "checklist_goal": (
                "Verify batch handoff completion handoff before opening official cpWER execution completion review."
            ),
            "bridge_note": (
                f"Batch handoff completion handoff reports handoff_status={handoff_status} with "
                f"queue_status={queue_status} at {complete_handoff_count}/{total_handoff_count} complete handoffs."
            ),
            "next_gate": "Confirm this bridge before opening the official cpWER execution completion summary.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval cpWER Execution Status Batch Handoff Completion Summary Handoff Bridge Checklist",
        "",
        "This generated checklist connects batch handoff completion handoff to official cpWER execution completion review. "
        "It does not claim official MeetEval evaluation or benchmark completion.",
        "",
        "| checklist_order | handoff_status | queue_status | complete_handoff_count | total_handoff_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['handoff_status']} | {row['queue_status']} | "
            f"{row['complete_handoff_count']} | {row['total_handoff_count']} | {row['prerequisite_artifact']} | "
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
        / "meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff_bridge_checklist.csv"
    )
    json_path = (
        tables_dir
        / "meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff_bridge_checklist.json"
    )
    md_path = (
        figures_dir
        / "meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff_bridge_checklist.md"
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_handoff_rows())
    if not rows:
        print("Batch handoff completion summary handoff not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER batch handoff completion handoff bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER batch handoff completion handoff bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER batch handoff completion handoff bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
