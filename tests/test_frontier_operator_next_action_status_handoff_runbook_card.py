from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_handoff_runbook_card import build_runbook_card_row


class FrontierOperatorNextActionStatusHandoffRunbookCardTest(unittest.TestCase):
    def test_build_runbook_card_row_targets_ready_frontier(self) -> None:
        row = build_runbook_card_row(
            {
                "ready_frontier": "meeteval_compatibility",
                "ready_action": "Fill the official receipt with real evidence.",
                "operator_evidence": "results/figures/frontier_operator_next_action_status_handoff.md",
                "operator_urgency": "queue_status=queue_complete; ready_lane_count=1; blocked_lane_count=1",
            },
            [
                {
                    "action_lane": "ready_lane",
                    "expected_outputs": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ],
        )

        self.assertEqual(row["recommended_frontier"], "meeteval_compatibility")
        self.assertIn("Fill the official receipt", row["recommended_action"])
        self.assertIn("meeteval_cpwer_execution_receipt.json", row["completion_signal"])

    def test_build_runbook_card_row_returns_empty_when_no_brief(self) -> None:
        row = build_runbook_card_row({}, [])

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
