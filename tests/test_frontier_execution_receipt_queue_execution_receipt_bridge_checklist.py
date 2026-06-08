from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_execution_receipt_bridge_checklist import (
    build_bridge_checklist_rows,
)


class FrontierExecutionReceiptQueueExecutionReceiptBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_execution_receipt_gate(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "receipt_frontier": "meeteval_compatibility",
                "prerequisite_artifact": "results/figures/frontier_execution_receipt_queue_evidence_receipt.md",
                "execution_receipt_target": "results/tables/meeteval_cpwer_execution_receipt.json",
                "bridge_note": "Verify evidence first.",
            }
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("meeteval_cpwer_execution_receipt.json", rows[0]["next_gate"])


if __name__ == "__main__":
    unittest.main()
