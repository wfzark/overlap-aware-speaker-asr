from __future__ import annotations

import unittest

from src.frontier_execution_receipt_fill_queue_status_bridge_checklist import build_bridge_checklist_rows


class FrontierExecutionReceiptFillQueueStatusBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_maps_fill_status(self) -> None:
        rows = build_bridge_checklist_rows(
            [
                {
                    "fill_order": "1",
                    "frontier_name": "meeteval_compatibility",
                    "fill_status": "awaiting_fill",
                    "receipt_path": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ]
        )

        self.assertEqual(rows[0]["frontier_name"], "meeteval_compatibility")
        self.assertEqual(rows[0]["fill_status"], "awaiting_fill")


if __name__ == "__main__":
    unittest.main()
