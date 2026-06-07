from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_completion_summary import build_completion_summary_row


class FrontierExecutionReceiptQueueCompletionSummaryTest(unittest.TestCase):
    def test_build_completion_summary_row_marks_queue_complete(self) -> None:
        row = build_completion_summary_row(
            {
                "meeteval_readiness_status": "receipt_ready_to_fill",
                "speaker_profile_readiness_status": "receipt_ready_to_fill",
                "external_staging_readiness_status": "receipt_ready_to_fill",
            }
        )

        self.assertEqual(row["queue_status"], "queue_complete")
        self.assertEqual(row["ready_receipt_count"], "3")


if __name__ == "__main__":
    unittest.main()
