from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_handoff_completion_dashboard import build_dashboard_row


class FrontierOperatorNextActionStatusHandoffCompletionDashboardTest(unittest.TestCase):
    def test_build_dashboard_row_summarizes_status_handoff_state(self) -> None:
        row = build_dashboard_row(
            {
                "ready_frontier": "meeteval_compatibility",
                "blocked_frontier": "external_validation",
            },
            {
                "next_milestone": "ready_lane_checkpoint_complete",
                "remaining_frontier_count": "1",
            },
        )

        self.assertEqual(row["current_first_frontier"], "meeteval_compatibility")
        self.assertEqual(row["blocked_frontier"], "external_validation")
        self.assertEqual(row["next_milestone"], "ready_lane_checkpoint_complete")
        self.assertEqual(row["dominant_blocker"], "external_validation")

    def test_build_dashboard_row_returns_empty_without_inputs(self) -> None:
        row = build_dashboard_row({}, {})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
