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
        "fill_queue_status",
        "results/figures/frontier_execution_receipt_fill_queue_status.md",
        "Shows per-frontier fill_status and execution_status without claiming benchmark execution.",
    ),
    (
        "2",
        "fill_queue_handoff",
        "results/figures/frontier_execution_receipt_fill_queue_handoff.md",
        "Turns fill queue status into per-frontier fill execution actions.",
    ),
    (
        "3",
        "fill_queue_handoff_bridge",
        "results/figures/frontier_execution_receipt_fill_queue_handoff_bridge_checklist.md",
        "Row-by-row bridge verification path before updating execution receipts.",
    ),
    (
        "4",
        "fill_queue_completion_summary",
        "results/figures/frontier_execution_receipt_fill_queue_completion_summary.md",
        "Rollup of awaiting_fill_count and combined_fill_status across all frontiers.",
    ),
]


def load_fill_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_fill_queue_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_packet_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    combined = str(summary.get("combined_fill_status", "fill_queue_empty"))
    awaiting = str(summary.get("awaiting_fill_count", "0"))
    rows: list[dict[str, str]] = []
    for order, section_name, artifact_path, section_role in PACKET_SECTIONS:
        rows.append(
            {
                "packet_order": order,
                "section_name": section_name,
                "artifact_path": artifact_path,
                "section_role": section_role,
                "packet_note": (
                    f"Fill execution packet section while combined_fill_status={combined} "
                    f"and awaiting_fill_count={awaiting}; no benchmark execution is claimed."
                ),
            }
        )
    return rows


def build_packet_lines(rows: list[dict[str, str]], summary: dict[str, str]) -> list[str]:
    combined = str(summary.get("combined_fill_status", "fill_queue_empty"))
    lines = [
        "# Frontier Execution Receipt Fill Execution Packet",
        "",
        "This generated note provides a single entrypoint for the receipt fill execution stack. "
        "It remains experimental/frontier coordination only and does not claim benchmark completion.",
        "",
        f"Current rollup: `combined_fill_status = {combined}`.",
        "",
        "| packet_order | section_name | artifact_path | section_role | packet_note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['packet_order']} | {row['section_name']} | {row['artifact_path']} | "
            f"{row['section_role']} | {row['packet_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]], summary: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_fill_execution_packet.csv"
    json_path = tables_dir / "frontier_execution_receipt_fill_execution_packet.json"
    md_path = figures_dir / "frontier_execution_receipt_fill_execution_packet.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PACKET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_packet_lines(rows, summary)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_fill_summary()
    rows = build_packet_rows(summary)
    csv_path, json_path, md_path = write_outputs(rows, summary)
    print(f"Wrote frontier execution receipt fill execution packet CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt fill execution packet JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt fill execution packet note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
