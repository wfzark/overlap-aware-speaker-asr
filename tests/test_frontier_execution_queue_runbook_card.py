from __future__ import annotations

import unittest

from src.frontier_execution_queue_runbook_card import build_runbook_card_row


class FrontierExecutionQueueRunbookCardTest(unittest.TestCase):
    def test_build_runbook_card_row_targets_first_frontier(self) -> None:
        row = build_runbook_card_row(
            {
                "operator_frontier": "meeteval_compatibility",
                "operator_action": "Fill the execution receipt at results/tables/meeteval_cpwer_execution_receipt.json after final bridge verification.",
                "operator_evidence": "results/figures/frontier_execution_queue_handoff.md",
                "operator_urgency": "queue_status=queue_complete; ready_chain_count=3; pending_chain_count=0",
            },
            [
                {
                    "frontier_name": "meeteval_compatibility",
                    "expected_outputs": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ],
        )

        self.assertEqual(row["recommended_frontier"], "meeteval_compatibility")
        self.assertIn("Fill the execution receipt", row["recommended_action"])
        self.assertIn("meeteval_cpwer_execution_receipt.json", row["completion_signal"])

    def test_build_runbook_card_row_returns_empty_when_no_brief(self) -> None:
        row = build_runbook_card_row({}, [])

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
