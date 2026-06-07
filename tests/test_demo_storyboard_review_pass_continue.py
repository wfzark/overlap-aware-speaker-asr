from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_continue import build_continue_row, select_next_card


class DemoStoryboardReviewPassContinueTest(unittest.TestCase):
    def test_select_next_card_skips_completed_cards(self) -> None:
        cards = [
            {"title": "Problem", "body": "a"},
            {"title": "Pipeline", "body": "b"},
            {"title": "Findings", "body": "c"},
        ]
        next_card, card_index = select_next_card(cards, {"Problem", "Pipeline"})

        self.assertEqual(next_card["title"], "Findings")
        self.assertEqual(card_index, 3)

    def test_build_continue_row_records_third_card(self) -> None:
        row = build_continue_row({"title": "Findings"}, 3, 2)

        self.assertEqual(row["card_title"], "Findings")
        self.assertEqual(row["completed_card_count"], "2")


if __name__ == "__main__":
    unittest.main()
