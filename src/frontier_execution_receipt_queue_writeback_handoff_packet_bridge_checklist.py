from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "combined_writeback_status",
    "awaiting_writeback_count",
    "writeback_complete_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_writeback_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_writeback_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    combined = str(summary.get("combined_writeback_status", "writeback_queue_empty"))
    awaiting = str(summary.get("awaiting_writeback_count", "0"))
    complete = str(summary.get("writeback_complete_count", "0"))
    return [
        {
            "checklist_order": "1",
            "combined_writeback_status": combined,
            "awaiting_writeback_count": awaiting,
            "writeback_complete_count": complete,
            "prerequisite_artifact": "results/figures/frontier_execution_receipt_queue_writeback_handoff_packet.md",
            "receipt_target": "results/figures/frontier_execution_receipt_queue_writeback_status.md",
            "checklist_goal": (
                "Verify the receipt queue writeback handoff packet before opening the writeback status rollup."
            ),
            "bridge_note": (
                f"Writeback summary reports combined_writeback_status={combined}, "
                f"awaiting_writeback_count={awaiting}, writeback_complete_count={complete}; "
                "confirm packet context before reopening the writeback status rollup."
            ),
            "next_gate": "Confirm this bridge before opening the frontier execution receipt queue writeback status target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Queue Writeback Handoff Packet Bridge Checklist",
        "",
        "This generated checklist turns the receipt queue writeback handoff packet into a bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim benchmark execution.",
        "",
        "| checklist_order | combined_writeback_status | awaiting_writeback_count | writeback_complete_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['combined_writeback_status']} | {row['awaiting_writeback_count']} | "
            f"{row['writeback_complete_count']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_writeback_handoff_packet_bridge_checklist.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_writeback_handoff_packet_bridge_checklist.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_writeback_handoff_packet_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_writeback_summary())
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote frontier execution receipt queue writeback handoff packet bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt queue writeback handoff packet bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt queue writeback handoff packet bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
