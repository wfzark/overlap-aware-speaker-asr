from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_runbook_card import build_runbook_card_row


class FrontierExecutionReceiptQueueRunbookCardTest(unittest.TestCase):
    def test_build_runbook_card_row_uses_operator_brief_frontier(self) -> None:
        row = build_runbook_card_row(
            {
                "operator_frontier": "meeteval_compatibility",
                "operator_action": (
                    "Update execution_status in results/tables/meeteval_cpwer_execution_receipt.json "
                    "after a real frontier run and bridge verification."
                ),
                "operator_evidence": (
                    "results/figures/frontier_execution_receipt_queue_handoff.md; "
                    "results/figures/frontier_execution_receipt_queue_handoff_bridge_checklist.md"
                ),
                "operator_urgency": "queue_status=queue_complete; ready_receipt_count=3; pending_receipt_count=0",
            },
            [
                {
                    "frontier_name": "meeteval_compatibility",
                    "expected_outputs": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ],
        )

        self.assertEqual(row["recommended_frontier"], "meeteval_compatibility")
        self.assertIn("meeteval_cpwer_execution_receipt.json", row["completion_signal"])
        self.assertIn("queue_complete", row["urgency"])

    def test_build_runbook_card_row_returns_empty_without_operator_brief(self) -> None:
        row = build_runbook_card_row({}, [])

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
