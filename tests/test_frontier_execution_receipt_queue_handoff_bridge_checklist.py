from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_handoff_bridge_checklist import build_bridge_checklist_rows


class FrontierExecutionReceiptQueueHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_maps_handoff_fields(self) -> None:
        rows = build_bridge_checklist_rows(
            [
                {
                    "handoff_order": "1",
                    "frontier_name": "meeteval_compatibility",
                    "readiness_status": "receipt_ready_to_fill",
                    "expected_outputs": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ]
        )

        self.assertEqual(rows[0]["frontier_name"], "meeteval_compatibility")
        self.assertIn("meeteval_cpwer_execution_receipt.json", rows[0]["receipt_target"])

    def test_build_bridge_checklist_rows_returns_empty_for_no_handoff(self) -> None:
        rows = build_bridge_checklist_rows([])

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
