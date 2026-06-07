from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_advance import build_advance_row, select_next_card


class DemoStoryboardReviewPassAdvanceTest(unittest.TestCase):
    def test_select_next_card_skips_completed_problem(self) -> None:
        cards = [
            {"title": "Problem", "body": "a"},
            {"title": "Pipeline", "body": "b"},
        ]
        next_card, card_index = select_next_card(cards, "Problem")

        self.assertEqual(next_card["title"], "Pipeline")
        self.assertEqual(card_index, 2)

    def test_build_advance_row_records_second_card(self) -> None:
        row = build_advance_row({"title": "Pipeline"}, 2, "Problem")

        self.assertEqual(row["card_title"], "Pipeline")
        self.assertEqual(row["review_order"], "2")


if __name__ == "__main__":
    unittest.main()
