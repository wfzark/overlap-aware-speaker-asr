from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_second_continue import build_continue_row, select_next_card


class DemoStoryboardReviewPassSecondContinueTest(unittest.TestCase):
    def test_select_next_card_skips_three_completed_cards(self) -> None:
        cards = [
            {"title": "Problem", "body": "a"},
            {"title": "Pipeline", "body": "b"},
            {"title": "Findings", "body": "c"},
            {"title": "Frontier", "body": "d"},
        ]
        next_card, card_index = select_next_card(cards, {"Problem", "Pipeline", "Findings"})

        self.assertEqual(next_card["title"], "Frontier")
        self.assertEqual(card_index, 4)

    def test_build_continue_row_records_fourth_card(self) -> None:
        row = build_continue_row({"title": "Frontier"}, 4, 3)

        self.assertEqual(row["card_title"], "Frontier")
        self.assertEqual(row["completed_card_count"], "3")


if __name__ == "__main__":
    unittest.main()
