from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "frontier_name",
    "writeback_status",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_handoff_rows() -> list[dict[str, str]]:
    handoff_path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_writeback_handoff.json"
    if not handoff_path.exists():
        return []
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(handoff_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for handoff in handoff_rows:
        order = str(handoff.get("handoff_order", len(rows) + 1))
        frontier_name = str(handoff.get("frontier_name", "unknown"))
        writeback_status = str(handoff.get("writeback_status", "receipt_missing"))
        receipt_target = str(handoff.get("expected_outputs", ""))
        rows.append(
            {
                "checklist_order": order,
                "frontier_name": frontier_name,
                "writeback_status": writeback_status,
                "prerequisite_artifact": "results/figures/frontier_execution_receipt_queue_writeback_handoff.md",
                "receipt_target": receipt_target,
                "checklist_goal": (
                    f"Verify the writeback handoff for {frontier_name} before updating the execution receipt."
                ),
                "bridge_note": (
                    f"Handoff reports writeback_status={writeback_status} for {frontier_name}; "
                    "confirm writeback context before writing back the execution receipt."
                ),
                "next_gate": f"Confirm this bridge before claiming any {frontier_name} benchmark execution.",
            }
        )
    return rows


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Queue Writeback Handoff Bridge Checklist",
        "",
        "This generated checklist turns the receipt queue writeback handoff into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim benchmark execution.",
        "",
        "| checklist_order | frontier_name | writeback_status | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['frontier_name']} | {row['writeback_status']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_writeback_handoff_bridge_checklist.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_writeback_handoff_bridge_checklist.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_writeback_handoff_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    handoff_rows = load_handoff_rows()
    rows = build_bridge_checklist_rows(handoff_rows)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote frontier execution receipt queue writeback handoff bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt queue writeback handoff bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt queue writeback handoff bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
