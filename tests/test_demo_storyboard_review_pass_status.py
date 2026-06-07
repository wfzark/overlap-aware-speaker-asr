from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_status import build_status_row


class DemoStoryboardReviewPassStatusTest(unittest.TestCase):
    def test_build_status_row_queue_complete_when_all_cards_done(self) -> None:
        cards = [{"title": "Problem"}, {"title": "Pipeline"}, {"title": "Findings"}, {"title": "Frontier"}]
        row = build_status_row(cards, {"Problem", "Pipeline", "Findings", "Frontier"})

        self.assertEqual(row["queue_status"], "queue_complete")
        self.assertEqual(row["pending_count"], "0")

    def test_build_status_row_queue_in_progress_when_cards_remain(self) -> None:
        cards = [{"title": "Problem"}, {"title": "Pipeline"}]
        row = build_status_row(cards, {"Problem"})

        self.assertEqual(row["queue_status"], "queue_in_progress")
        self.assertEqual(row["pending_count"], "1")


if __name__ == "__main__":
    unittest.main()
