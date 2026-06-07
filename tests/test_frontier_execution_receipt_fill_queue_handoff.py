from __future__ import annotations

import unittest

from src.frontier_execution_receipt_fill_queue_handoff import build_handoff_rows


class FrontierExecutionReceiptFillQueueHandoffTest(unittest.TestCase):
    def test_build_handoff_rows_recommends_fill_when_awaiting(self) -> None:
        rows = build_handoff_rows(
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
        self.assertIn("Fill execution_status", rows[0]["recommended_action"])


if __name__ == "__main__":
    unittest.main()
