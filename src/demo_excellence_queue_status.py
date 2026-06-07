from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


STATUS_COLUMNS = [
    "scope",
    "walkthrough_queue_status",
    "storyboard_queue_status",
    "combined_queue_status",
    "status_note",
]


def load_summary(path_rel: str) -> dict[str, str]:
    summary_path = PROJECT_ROOT / path_rel
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_status_row(
    walkthrough_summary: dict[str, str],
    storyboard_summary: dict[str, str],
) -> dict[str, str]:
    walkthrough_status = str(walkthrough_summary.get("queue_status", "queue_in_progress"))
    storyboard_status = str(storyboard_summary.get("queue_status", "queue_in_progress"))
    combined_status = (
        "queue_complete"
        if walkthrough_status == "queue_complete" and storyboard_status == "queue_complete"
        else "queue_in_progress"
    )
    return {
        "scope": "demo_excellence_review_queues",
        "walkthrough_queue_status": walkthrough_status,
        "storyboard_queue_status": storyboard_status,
        "combined_queue_status": combined_status,
        "status_note": (
            "Unified qualitative/demo review queue rollup across walkthrough and storyboard passes; "
            "no live demo or recording delivery is claimed."
        ),
    }


def build_status_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Excellence Queue Status",
        "",
        "This generated note records the unified demo excellence review queue rollup. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| scope | walkthrough_queue_status | storyboard_queue_status | combined_queue_status | status_note |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| {row['scope']} | {row['walkthrough_queue_status']} | {row['storyboard_queue_status']} | "
            f"{row['combined_queue_status']} | {row['status_note']} |"
        ),
    ]
    return lines


def write_outputs(status_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "demo_excellence_queue_status.csv"
    json_path = tables_dir / "demo_excellence_queue_status.json"
    md_path = figures_dir / "demo_excellence_queue_status.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerow(status_row)
    json_path.write_text(json.dumps(status_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_status_lines(status_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    walkthrough_summary = load_summary("results/tables/demo_walkthrough_review_pass_completion_summary.json")
    storyboard_summary = load_summary("results/tables/demo_storyboard_review_pass_completion_summary.json")
    status_row = build_status_row(walkthrough_summary, storyboard_summary)
    csv_path, json_path, md_path = write_outputs(status_row)
    print(f"Wrote demo excellence queue status CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo excellence queue status JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo excellence queue status note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Combined queue status: {status_row['combined_queue_status']}")


if __name__ == "__main__":
    main()
