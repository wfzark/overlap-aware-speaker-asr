from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


PACKET_COLUMNS = [
    "packet_order",
    "section_name",
    "artifact_path",
    "section_role",
    "packet_note",
]


PACKET_SECTIONS = [
    (
        "1",
        "receipt_queue_writeback_status",
        "results/figures/frontier_execution_receipt_queue_writeback_status.md",
        "Dynamic writeback rollup across the frontier execution receipts.",
    ),
    (
        "2",
        "receipt_queue_writeback_handoff",
        "results/figures/frontier_execution_receipt_queue_writeback_handoff.md",
        "Per-frontier writeback next actions derived from the current mixed status state.",
    ),
    (
        "3",
        "receipt_queue_writeback_handoff_bridge_checklist",
        "results/figures/frontier_execution_receipt_queue_writeback_handoff_bridge_checklist.md",
        "Row-by-row gate before reopening any frontier receipt for writeback.",
    ),
    (
        "4",
        "receipt_queue_writeback_open_card",
        "results/figures/frontier_execution_receipt_queue_writeback_open_card.md",
        "Single current writeback target card, skipping already-complete receipts.",
    ),
    (
        "5",
        "receipt_queue_writeback_open_card_bridge_checklist",
        "results/figures/frontier_execution_receipt_queue_writeback_open_card_bridge_checklist.md",
        "Gate before reopening the currently selected execution receipt target.",
    ),
]


def load_writeback_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_writeback_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_packet_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    combined = str(summary.get("combined_writeback_status", "writeback_queue_empty"))
    awaiting = str(summary.get("awaiting_writeback_count", "0"))
    complete = str(summary.get("writeback_complete_count", "0"))
    rows: list[dict[str, str]] = []
    for order, section_name, artifact_path, section_role in PACKET_SECTIONS:
        rows.append(
            {
                "packet_order": order,
                "section_name": section_name,
                "artifact_path": artifact_path,
                "section_role": section_role,
                "packet_note": (
                    f"Writeback packet section while combined_writeback_status={combined}, "
                    f"awaiting_writeback_count={awaiting}, writeback_complete_count={complete}; "
                    "no benchmark execution is claimed beyond receipt contents."
                ),
            }
        )
    return rows


def build_packet_lines(rows: list[dict[str, str]], summary: dict[str, str]) -> list[str]:
    combined = str(summary.get("combined_writeback_status", "writeback_queue_empty"))
    return [
        "# Frontier Execution Receipt Queue Writeback Handoff Packet",
        "",
        "This generated note provides a compact entrypoint for the receipt-queue writeback handoff stack. "
        "It remains experimental/frontier coordination only and does not claim benchmark completion.",
        "",
        f"Current rollup: `combined_writeback_status = {combined}`.",
        "",
        "| packet_order | section_name | artifact_path | section_role | packet_note |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {row['packet_order']} | {row['section_name']} | {row['artifact_path']} | "
            f"{row['section_role']} | {row['packet_note']} |"
            for row in rows
        ],
    ]


def write_outputs(rows: list[dict[str, str]], summary: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_writeback_handoff_packet.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_writeback_handoff_packet.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_writeback_handoff_packet.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PACKET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_packet_lines(rows, summary)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_writeback_summary()
    rows = build_packet_rows(summary)
    csv_path, json_path, md_path = write_outputs(rows, summary)
    print(
        "Wrote frontier execution receipt queue writeback handoff packet CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt queue writeback handoff packet JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier execution receipt queue writeback handoff packet note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
