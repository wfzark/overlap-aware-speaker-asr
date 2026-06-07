from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "frontier_name",
    "fill_status",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_fill_status_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_fill_queue_status.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(fill_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fill_row in fill_rows:
        order = str(fill_row.get("fill_order", len(rows) + 1))
        frontier_name = str(fill_row.get("frontier_name", "unknown"))
        fill_status = str(fill_row.get("fill_status", "fill_blocked"))
        receipt_target = str(fill_row.get("receipt_path", ""))
        rows.append(
            {
                "checklist_order": order,
                "frontier_name": frontier_name,
                "fill_status": fill_status,
                "prerequisite_artifact": "results/figures/frontier_execution_receipt_fill_queue_status.md",
                "receipt_target": receipt_target,
                "checklist_goal": (
                    f"Verify the receipt fill status for {frontier_name} before updating the execution receipt."
                ),
                "bridge_note": (
                    f"Fill queue reports fill_status={fill_status} for {frontier_name}; "
                    "confirm fill context before writing back the execution receipt."
                ),
                "next_gate": f"Confirm this bridge before claiming any {frontier_name} benchmark execution.",
            }
        )
    return rows


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Fill Queue Status Bridge Checklist",
        "",
        "This generated checklist turns the receipt fill queue status into a row-by-row bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim benchmark execution.",
        "",
        "| checklist_order | frontier_name | fill_status | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['frontier_name']} | {row['fill_status']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_fill_queue_status_bridge_checklist.csv"
    json_path = tables_dir / "frontier_execution_receipt_fill_queue_status_bridge_checklist.json"
    md_path = figures_dir / "frontier_execution_receipt_fill_queue_status_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    fill_rows = load_fill_status_rows()
    rows = build_bridge_checklist_rows(fill_rows)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote frontier execution receipt fill queue status bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt fill queue status bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt fill queue status bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
