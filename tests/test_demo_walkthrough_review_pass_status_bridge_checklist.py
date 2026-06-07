from __future__ import annotations

import unittest

from src.demo_walkthrough_review_pass_status_bridge_checklist import build_bridge_checklist_rows


class DemoWalkthroughReviewPassStatusBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_final_bridge(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "queue_status": "queue_complete",
                "completed_count": "5",
            }
        )

        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertIn("final_bridge", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
