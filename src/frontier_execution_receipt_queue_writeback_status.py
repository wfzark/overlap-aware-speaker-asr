from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .frontier_execution_receipt_fill_queue_status import (
    TEMPLATE_STATUSES,
    load_receipt_execution_status,
)


WRITEBACK_STATUS_COLUMNS = [
    "writeback_order",
    "frontier_name",
    "receipt_path",
    "execution_status",
    "readiness_status",
    "writeback_status",
    "writeback_note",
]


def load_handoff_rows() -> list[dict[str, str]]:
    handoff_path = PROJECT_ROOT / "results" / "tables" / "frontier_execution_receipt_queue_handoff.json"
    if not handoff_path.exists():
        return []
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def derive_writeback_status(execution_status: str, readiness_status: str) -> str:
    if execution_status == "missing":
        return "receipt_missing"
    if readiness_status != "receipt_ready_to_fill":
        return "writeback_blocked"
    if execution_status in TEMPLATE_STATUSES:
        return "awaiting_writeback"
    return "writeback_complete"


def build_writeback_status_rows(handoff_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for handoff in handoff_rows:
        order = str(handoff.get("handoff_order", len(rows) + 1))
        frontier_name = str(handoff.get("frontier_name", "unknown"))
        readiness_status = str(handoff.get("readiness_status", "receipt_not_ready"))
        receipt_path = str(handoff.get("expected_outputs", ""))
        execution_status = load_receipt_execution_status(receipt_path) if receipt_path else "missing"
        writeback_status = derive_writeback_status(execution_status, readiness_status)
        rows.append(
            {
                "writeback_order": order,
                "frontier_name": frontier_name,
                "receipt_path": receipt_path,
                "execution_status": execution_status,
                "readiness_status": readiness_status,
                "writeback_status": writeback_status,
                "writeback_note": (
                    f"Receipt queue writeback status for {frontier_name} while execution_status={execution_status}; "
                    "no benchmark execution is claimed until receipt evidence is actually written back."
                ),
            }
        )
    return rows


def build_status_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    awaiting = sum(1 for row in rows if row["writeback_status"] == "awaiting_writeback")
    blocked = sum(1 for row in rows if row["writeback_status"] == "writeback_blocked")
    complete = sum(1 for row in rows if row["writeback_status"] == "writeback_complete")
    total = len(rows)
    if total == 0:
        combined = "writeback_queue_empty"
    elif awaiting == total:
        combined = "writeback_queue_ready"
    elif complete == total:
        combined = "writeback_queue_complete"
    else:
        combined = "writeback_queue_in_progress"
    return {
        "scope": "frontier_execution_receipt_queue_writeback",
        "total_frontier_count": str(total),
        "awaiting_writeback_count": str(awaiting),
        "writeback_blocked_count": str(blocked),
        "writeback_complete_count": str(complete),
        "combined_writeback_status": combined,
        "writeback_note": (
            "experimental/frontier receipt-queue writeback rollup; "
            "template-only receipts remain unfilled and filled receipts stay explicitly labeled."
        ),
    }


def build_writeback_status_lines(rows: list[dict[str, str]], summary: dict[str, str]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Queue Writeback Status",
        "",
        "This generated note rolls up receipt-queue writeback status across the frontier execution receipts. "
        "It remains experimental/frontier coordination only and does not claim benchmark completion.",
        "",
        "## Summary",
        "",
        f"- combined_writeback_status: `{summary['combined_writeback_status']}`",
        f"- awaiting_writeback_count: `{summary['awaiting_writeback_count']}/{summary['total_frontier_count']}`",
        f"- writeback_complete_count: `{summary['writeback_complete_count']}/{summary['total_frontier_count']}`",
        "",
        "| writeback_order | frontier_name | receipt_path | execution_status | readiness_status | writeback_status | writeback_note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['writeback_order']} | {row['frontier_name']} | {row['receipt_path']} | "
            f"{row['execution_status']} | {row['readiness_status']} | {row['writeback_status']} | {row['writeback_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]], summary: dict[str, str]) -> tuple[Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_writeback_status.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_writeback_status.json"
    summary_path = tables_dir / "frontier_execution_receipt_queue_writeback_summary.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_writeback_status.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=WRITEBACK_STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_writeback_status_lines(rows, summary)) + "\n", encoding="utf-8")
    return csv_path, json_path, summary_path, md_path


def main() -> None:
    handoff_rows = load_handoff_rows()
    rows = build_writeback_status_rows(handoff_rows)
    summary = build_status_summary(rows)
    csv_path, json_path, summary_path, md_path = write_outputs(rows, summary)
    print(f"Wrote frontier execution receipt queue writeback status CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue writeback status JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue writeback summary JSON: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue writeback status note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Combined writeback status: {summary['combined_writeback_status']}")


if __name__ == "__main__":
    main()
