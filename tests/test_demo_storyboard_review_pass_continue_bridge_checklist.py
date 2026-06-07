from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_continue_bridge_checklist import build_bridge_checklist_rows


class DemoStoryboardReviewPassContinueBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_fourth_card(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "card_index": "3",
                "card_title": "Findings",
                "review_status": "review_complete",
            }
        )

        self.assertEqual(rows[0]["card_title"], "Findings")
        self.assertIn("second continue", rows[0]["next_gate"].lower())


if __name__ == "__main__":
    unittest.main()
