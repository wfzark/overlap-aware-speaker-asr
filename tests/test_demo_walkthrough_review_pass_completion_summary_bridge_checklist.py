from __future__ import annotations

import unittest

from src.demo_walkthrough_review_pass_completion_summary_bridge_checklist import build_bridge_checklist_rows


class DemoWalkthroughReviewPassCompletionSummaryBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_storyboard_review(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "queue_status": "queue_complete",
                "completed_count": "5",
            }
        )

        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertIn("storyboard_review_pass", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
