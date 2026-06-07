from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .demo_storyboard_review_pass import load_storyboard_cards


STATUS_COLUMNS = [
    "queue_status",
    "completed_count",
    "total_card_count",
    "pending_count",
    "status_note",
]

COMPLETED_REVIEW_PATHS = (
    "results/tables/demo_storyboard_review_pass.json",
    "results/tables/demo_storyboard_review_pass_second.json",
    "results/tables/demo_storyboard_review_pass_third.json",
    "results/tables/demo_storyboard_review_pass_fourth.json",
)


def load_completed_card_titles() -> set[str]:
    completed: set[str] = set()
    for rel_path in COMPLETED_REVIEW_PATHS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            card_title = str(payload.get("card_title", "")).strip()
            if card_title:
                completed.add(card_title)
    return completed


def build_status_row(cards: list[dict[str, str]], completed_titles: set[str]) -> dict[str, str]:
    total_card_count = len(cards)
    completed_count = len(completed_titles)
    pending_count = max(0, total_card_count - completed_count)
    queue_status = "queue_complete" if pending_count == 0 else "queue_in_progress"
    return {
        "queue_status": queue_status,
        "completed_count": str(completed_count),
        "total_card_count": str(total_card_count),
        "pending_count": str(pending_count),
        "status_note": (
            f"Storyboard review queue at {completed_count}/{total_card_count}; "
            "no live demo or recording delivery is claimed."
        ),
    }


def build_status_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Storyboard Review Pass Status",
        "",
        "This generated note records the storyboard review queue rollup. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| queue_status | completed_count | total_card_count | pending_count | status_note |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            f"| {row['queue_status']} | {row['completed_count']} | {row['total_card_count']} | "
            f"{row['pending_count']} | {row['status_note']} |"
        ),
    ]
    return lines


def write_outputs(status_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "demo_storyboard_review_pass_status.csv"
    json_path = tables_dir / "demo_storyboard_review_pass_status.json"
    md_path = figures_dir / "demo_storyboard_review_pass_status.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerow(status_row)
    json_path.write_text(json.dumps(status_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_status_lines(status_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    cards = load_storyboard_cards()
    completed_titles = load_completed_card_titles()
    status_row = build_status_row(cards, completed_titles)
    csv_path, json_path, md_path = write_outputs(status_row)
    print(f"Wrote demo storyboard review pass status CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass status JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass status note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Queue status: {status_row['queue_status']}")


if __name__ == "__main__":
    main()
