from __future__ import annotations

import unittest

from src.frontier_operator_next_action_completion_dashboard import build_dashboard_row


class FrontierOperatorNextActionCompletionDashboardTest(unittest.TestCase):
    def test_build_dashboard_row_summarizes_top_level_state(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
