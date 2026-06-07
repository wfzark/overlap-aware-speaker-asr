from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_completion_summary import build_completion_summary_row
from src.demo_storyboard_review_pass_status_bridge_checklist import build_bridge_checklist_rows


class DemoStoryboardReviewPassStatusBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_completion_summary(self) -> None:
        rows = build_bridge_checklist_rows(
            {"queue_status": "queue_complete", "completed_count": "4"}
        )

        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertIn("completion summary", rows[0]["next_gate"].lower())


class DemoStoryboardReviewPassCompletionSummaryTest(unittest.TestCase):
    def test_build_completion_summary_row_queue_complete(self) -> None:
        row = build_completion_summary_row(
            {
                "completed_count": "4",
                "total_card_count": "4",
                "pending_count": "0",
            }
        )

        self.assertEqual(row["queue_status"], "queue_complete")
        self.assertEqual(row["scope"], "storyboard_review_queue")


if __name__ == "__main__":
    unittest.main()
