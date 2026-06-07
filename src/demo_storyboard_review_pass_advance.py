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


ADVANCE_COLUMNS = [
    "review_order",
    "card_index",
    "prior_card_status",
    "card_title",
    "advance_note",
]


def load_completed_review_card() -> str:
    review_path = PROJECT_ROOT / "results" / "tables" / "demo_storyboard_review_pass.json"
    if not review_path.exists():
        return ""
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return str(payload.get("card_title", ""))
    return ""


def select_next_card(cards: list[dict[str, str]], completed_card_title: str) -> tuple[dict[str, str], int]:
    for index, card in enumerate(cards):
        title = str(card.get("title", ""))
        if title and title != completed_card_title:
            return card, index + 1
        if index == 0 and not completed_card_title:
            return card, index + 1
    if len(cards) > 1:
        return cards[1], 2
    return {"title": "Pipeline", "body": ""}, 2


def build_advance_row(next_card: dict[str, str], card_index: int, completed_card_title: str) -> dict[str, str]:
    card_title = str(next_card.get("title", "Pipeline"))
    return {
        "review_order": "2",
        "card_index": str(card_index),
        "prior_card_status": f"{completed_card_title or 'Problem'} review_complete",
        "card_title": card_title,
        "advance_note": (
            f"Storyboard review advanced to card {card_index} ({card_title}); "
            "no live demo or recording is claimed."
        ),
    }


def build_advance_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Demo Storyboard Review Pass Advance",
        "",
        "This generated note records the second qualitative storyboard review pass in card order. "
        "It remains qualitative/demo support only and does not claim a live demo or recording.",
        "",
        "| review_order | card_index | prior_card_status | card_title | advance_note |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| {row['review_order']} | {row['card_index']} | {row['prior_card_status']} | {row['card_title']} | "
            f"{row['advance_note']} |"
        ),
    ]
    return lines


def write_outputs(
    advance_row: dict[str, str],
    review_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    advance_csv_path = tables_dir / "demo_storyboard_review_pass_advance.csv"
    advance_json_path = tables_dir / "demo_storyboard_review_pass_advance.json"
    advance_md_path = figures_dir / "demo_storyboard_review_pass_advance.md"
    second_csv_path = tables_dir / "demo_storyboard_review_pass_second.csv"
    second_json_path = tables_dir / "demo_storyboard_review_pass_second.json"
    second_md_path = figures_dir / "demo_storyboard_review_pass_second.md"
    receipt_json_path = tables_dir / "demo_storyboard_review_pass_advance_receipt.json"
    receipt_md_path = figures_dir / "demo_storyboard_review_pass_advance_receipt.md"

    with advance_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ADVANCE_COLUMNS)
        writer.writeheader()
        writer.writerow(advance_row)
    advance_json_path.write_text(json.dumps(advance_row, ensure_ascii=False, indent=2), encoding="utf-8")
    advance_md_path.write_text("\n".join(build_advance_lines(advance_row)) + "\n", encoding="utf-8")

    second_review_row = dict(review_row)
    second_review_row["review_order"] = "2"
    with second_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(second_review_row)
    second_json_path.write_text(json.dumps(second_review_row, ensure_ascii=False, indent=2), encoding="utf-8")
    second_md_path.write_text("\n".join(build_review_lines(second_review_row)) + "\n", encoding="utf-8")

    receipt_json_path.write_text(json.dumps(receipt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt_md_path.write_text("\n".join(build_review_receipt_lines(receipt_rows)) + "\n", encoding="utf-8")
    return (
        advance_csv_path,
        advance_json_path,
        advance_md_path,
        second_csv_path,
        second_json_path,
        second_md_path,
        receipt_json_path,
        receipt_md_path,
    )


def main() -> None:
    cards = load_storyboard_cards()
    completed_card_title = load_completed_review_card()
    next_card, card_index = select_next_card(cards, completed_card_title)
    review_row = build_review_row(next_card, card_index=card_index)
    advance_row = build_advance_row(next_card, card_index, completed_card_title)
    receipt_rows = build_review_receipt_rows(review_row, len(cards))
    for receipt in receipt_rows:
        receipt["execution_status"] = "review_complete"
        receipt["review_scope"] = "second_storyboard_card"
        receipt["writeback_note"] = (
            f"Second qualitative storyboard review documented for card {card_index}; "
            "live demo or recording delivery remains pending."
        )
    (
        advance_csv_path,
        advance_json_path,
        advance_md_path,
        second_csv_path,
        second_json_path,
        second_md_path,
        receipt_json_path,
        receipt_md_path,
    ) = write_outputs(advance_row, review_row, receipt_rows)
    print(f"Wrote demo storyboard review pass advance CSV: {advance_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass advance JSON: {advance_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass advance note: {advance_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass second CSV: {second_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass second JSON: {second_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass second note: {second_md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass advance receipt JSON: {receipt_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard review pass advance receipt note: {receipt_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
