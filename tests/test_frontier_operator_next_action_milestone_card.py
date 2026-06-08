from __future__ import annotations

import unittest

from src.frontier_operator_next_action_milestone_card import build_milestone_card_row


class FrontierOperatorNextActionMilestoneCardTest(unittest.TestCase):
    def test_build_milestone_card_row_unlocks_blocked_frontier(self) -> None:
        row = build_milestone_card_row(
            {
                "ready_frontier": "meeteval_compatibility",
                "blocked_frontier": "external_validation",
            }
        )

        self.assertEqual(row["next_milestone"], "ready_lane_checkpoint_complete")
        self.assertIn("external_validation", row["unlocks"])
        self.assertEqual(row["remaining_frontier_count"], "1")


if __name__ == "__main__":
    unittest.main()
