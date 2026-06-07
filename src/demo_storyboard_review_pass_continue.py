from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .demo_storyboard_review_pass import (
    REVIEW_COLUMNS,
    build_review_lines,
    build_review_receipt_lines,
    build_review_receipt_rows,
    build_review_row,
    load_storyboard_cards,
)


CONTINUE_COLUMNS = [
    "review_order",
    "card_index",
    "completed_card_count",
    "card_title",
    "continue_note",
]

COMPLETED_REVIEW_PATHS = (
    "results/tables/demo_storyboard_review_pass.json",
    "results/tables/demo_storyboard_review_pass_second.json",
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


def select_next_card(cards: list[dict[str, str]], completed_titles: set[str]) -> tuple[dict[str, str], int]:
    for index, card in enumerate(cards):
        title = str(card.get("title", ""))
        if title and title not in completed_titles:
            return card, index + 1
    return cards[2] if len(cards) > 2 else {"title": "Findings", "body": ""}, 3


def build_continue_row(next_card: dict[str, str], card_index: int, completed_count: int) -> dict[str, str]:
    card_title = str(next_card.get("title", "Findings"))
    return {
        "review_order": "3",
        "card_index": str(card_index),
        "completed_card_count": str(completed_count),
        "card_title": card_title,
        "continue_note": (
            f"Storyboard review continued to card {card_index} ({card_title}); "
            "no live demo or recording is claimed."
        ),
    }


def build_continue_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Storyboard Review Pass Continue",
        "",
        "This generated note records the third qualitative storyboard review pass in card order. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| review_order | card_index | completed_card_count | card_title | continue_note |",
        "| --- | --- | ---: | --- | --- |",
        (
            f"| {row['review_order']} | {row['card_index']} | {row['completed_card_count']} | {row['card_title']} | "
            f"{row['continue_note']} |"
        ),
    ]
    return lines


def write_outputs(
    continue_row: dict[str, str],
    review_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    continue_csv_path = tables_dir / "demo_storyboard_review_pass_continue.csv"
    continue_json_path = tables_dir / "demo_storyboard_review_pass_continue.json"
    continue_md_path = figures_dir / "demo_storyboard_review_pass_continue.md"
    third_csv_path = tables_dir / "demo_storyboard_review_pass_third.csv"
    third_json_path = tables_dir / "demo_storyboard_review_pass_third.json"
    third_md_path = figures_dir / "demo_storyboard_review_pass_third.md"
    receipt_json_path = tables_dir / "demo_storyboard_review_pass_continue_receipt.json"
    receipt_md_path = figures_dir / "demo_storyboard_review_pass_continue_receipt.md"

    with continue_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CONTINUE_COLUMNS)
        writer.writeheader()
        writer.writerow(continue_row)
    continue_json_path.write_text(json.dumps(continue_row, ensure_ascii=False, indent=2), encoding="utf-8")
    continue_md_path.write_text("\n".join(build_continue_lines(continue_row)) + "\n", encoding="utf-8")

    third_review_row = dict(review_row)
    third_review_row["review_order"] = "3"
    with third_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(third_review_row)
    third_json_path.write_text(json.dumps(third_review_row, ensure_ascii=False, indent=2), encoding="utf-8")
    third_md_path.write_text("\n".join(build_review_lines(third_review_row)) + "\n", encoding="utf-8")

    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_review_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return (
        continue_csv_path,
        continue_json_path,
        continue_md_path,
        third_csv_path,
        third_json_path,
        third_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def main() -> None:
    cards = load_storyboard_cards()
    completed_titles = load_completed_card_titles()
    next_card, card_index = select_next_card(cards, completed_titles)
    review_row = build_review_row(next_card, card_index=card_index)
    continue_row = build_continue_row(next_card, card_index, len(completed_titles))
    receipt_rows = build_review_receipt_rows(review_row, len(cards))
    for receipt in receipt_rows:
        receipt["execution_status"] = "review_complete"
        receipt["review_scope"] = "third_storyboard_card"
        receipt["writeback_note"] = (
            f"Third qualitative storyboard review documented for card {card_index}; "
            "live demo or recording delivery remains pending."
        )
    (
        continue_csv_path,
        continue_json_path,
        continue_md_path,
        third_csv_path,
        third_json_path,
        third_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(continue_row, review_row, receipt_rows)
    print(f"Wrote demo storyboard review pass continue CSV: {continue_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass continue JSON: {continue_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass continue note: {continue_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass third CSV: {third_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass third JSON: {third_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass third note: {third_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass continue receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass continue receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
