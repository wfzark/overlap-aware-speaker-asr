from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_handoff_completion_summary_bridge_checklist import (
    build_bridge_checklist_rows,
)


class FrontierOperatorNextActionStatusHandoffCompletionSummaryBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_uses_queue_status_and_counts(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "queue_status": "queue_complete",
                "ready_lane_count": "1",
                "blocked_lane_count": "1",
            }
        )

        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertEqual(rows[0]["ready_lane_count"], "1")
        self.assertIn("blocked_lane_count=1", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_defaults_when_missing(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows[0]["queue_status"], "queue_empty")


if __name__ == "__main__":
    unittest.main()
