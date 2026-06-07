from __future__ import annotations

import unittest

from src.frontier_execution_receipt_fill_queue_completion_summary import build_completion_summary_row


class FrontierExecutionReceiptFillQueueCompletionSummaryTest(unittest.TestCase):
    def test_build_completion_summary_row_uses_fill_queue_ready(self) -> None:
        row = build_completion_summary_row(
            {
                "awaiting_fill_count": "3",
                "total_frontier_count": "3",
                "fill_complete_count": "0",
                "combined_fill_status": "fill_queue_ready",
            }
        )

        self.assertEqual(row["combined_fill_status"], "fill_queue_ready")
        self.assertEqual(row["awaiting_fill_count"], "3")


if __name__ == "__main__":
    unittest.main()
