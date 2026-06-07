from __future__ import annotations

import unittest

from src.demo_walkthrough_review_pass_continue_bridge_checklist import build_bridge_checklist_rows


class DemoWalkthroughReviewPassContinueBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_second_continue(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "step_id": "3",
                "review_status": "review_complete",
                "focus": "Routing takeaway",
            }
        )

        self.assertEqual(rows[0]["step_id"], "3")
        self.assertIn("second_continue", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
