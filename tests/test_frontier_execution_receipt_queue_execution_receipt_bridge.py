from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_execution_receipt_bridge import (
    build_execution_receipt_bridge_row,
)


class FrontierExecutionReceiptQueueExecutionReceiptBridgeTest(unittest.TestCase):
    def test_build_execution_receipt_bridge_row_targets_json_receipt(self) -> None:
        row = build_execution_receipt_bridge_row(
            {"receipt_frontier": "meeteval_compatibility"},
            {"operator_receipt": "results/tables/meeteval_cpwer_execution_receipt.json"},
        )

        self.assertEqual(row["receipt_frontier"], "meeteval_compatibility")
        self.assertIn("evidence_receipt", row["prerequisite_artifact"])
        self.assertIn("meeteval_cpwer_execution_receipt.json", row["execution_receipt_target"])


if __name__ == "__main__":
    unittest.main()
