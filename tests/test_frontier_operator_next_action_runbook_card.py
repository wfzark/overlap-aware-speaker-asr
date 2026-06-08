from __future__ import annotations

import unittest

from src.frontier_operator_next_action_runbook_card import build_runbook_card_row


class FrontierOperatorNextActionRunbookCardTest(unittest.TestCase):
    def test_build_runbook_card_row_promotes_ready_frontier(self) -> None:
        row = build_runbook_card_row(
            {
                "ready_frontier": "meeteval_compatibility",
                "ready_action": "Fill the official receipt with real evidence.",
                "operator_evidence": "results/figures/frontier_operator_next_action_card.md",
                "operator_urgency": "coordination_state=mixed_ready_state; active_lanes=2",
            },
            [
                {
                    "action_lane": "ready_lane",
                    "frontier_name": "meeteval_compatibility",
                    "target_artifact": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ],
        )

        self.assertEqual(row["recommended_frontier"], "meeteval_compatibility")
        self.assertIn("Fill the official receipt", row["recommended_action"])
        self.assertIn("active_lanes=2", row["urgency"])

    def test_build_runbook_card_row_returns_empty_when_no_brief(self) -> None:
        row = build_runbook_card_row({}, [])

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
