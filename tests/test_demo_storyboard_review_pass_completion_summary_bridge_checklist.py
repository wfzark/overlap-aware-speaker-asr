from __future__ import annotations

import unittest

from src.demo_storyboard_review_pass_completion_summary_bridge_checklist import build_bridge_checklist_rows


class DemoStoryboardReviewPassCompletionSummaryBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_demo_excellence_status(self) -> None:
        rows = build_bridge_checklist_rows({"queue_status": "queue_complete", "completed_count": "4"})

        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertIn("demo excellence", rows[0]["next_gate"].lower())


if __name__ == "__main__":
    unittest.main()
