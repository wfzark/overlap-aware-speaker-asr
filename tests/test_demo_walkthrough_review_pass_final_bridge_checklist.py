from __future__ import annotations

import unittest

from src.demo_walkthrough_review_pass_final_bridge_checklist import build_bridge_checklist_rows


class DemoWalkthroughReviewPassFinalBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_completion_summary(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "step_id": "5",
                "focus": "Next-step framing",
            }
        )

        self.assertEqual(rows[0]["step_id"], "5")
        self.assertIn("completion_summary", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
