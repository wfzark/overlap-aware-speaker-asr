from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


HANDOFF_COLUMNS = [
    "handoff_order",
    "frontier_name",
    "fill_status",
    "recommended_action",
    "expected_inputs",
    "expected_outputs",
    "handoff_note",
]


def load_fill_status_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_fill_queue_status.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_handoff_rows(fill_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fill_row in fill_rows:
        order = str(fill_row.get("fill_order", len(rows) + 1))
        frontier_name = str(fill_row.get("frontier_name", "unknown"))
        fill_status = str(fill_row.get("fill_status", "fill_blocked"))
        receipt_path = str(fill_row.get("receipt_path", ""))
        if fill_status == "awaiting_fill":
            action = (
                f"Fill execution_status in {receipt_path} after a real frontier run and bridge verification."
            )
        elif fill_status == "fill_complete":
            action = f"Review the filled receipt at {receipt_path} and archive the fill evidence note."
        else:
            action = "Resolve fill blockers before opening the execution receipt."
        rows.append(
            {
                "handoff_order": order,
                "frontier_name": frontier_name,
                "fill_status": fill_status,
                "recommended_action": action,
                "expected_inputs": (
                    "results/figures/frontier_execution_receipt_fill_queue_status.md; "
                    "per-frontier fill queue status bridge checklist."
                ),
                "expected_outputs": receipt_path,
                "handoff_note": (
                    f"Receipt-fill execution handoff for {frontier_name} while fill_status={fill_status}; "
                    "no benchmark execution or external audio staging is claimed."
                ),
            }
        )
    return rows


def build_handoff_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Fill Queue Handoff",
        "",
        "This generated note turns the receipt fill queue status into per-frontier fill execution actions. "
        "It remains experimental/frontier coordination only and does not claim benchmark completion.",
        "",
        "| handoff_order | frontier_name | fill_status | recommended_action | expected_inputs | expected_outputs | handoff_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['handoff_order']} | {row['frontier_name']} | {row['fill_status']} | "
            f"{row['recommended_action']} | {row['expected_inputs']} | {row['expected_outputs']} | {row['handoff_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_fill_queue_handoff.csv"
    json_path = tables_dir / "frontier_execution_receipt_fill_queue_handoff.json"
    md_path = figures_dir / "frontier_execution_receipt_fill_queue_handoff.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_handoff_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_handoff_rows(load_fill_status_rows())
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote frontier execution receipt fill queue handoff CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt fill queue handoff JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt fill queue handoff note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
