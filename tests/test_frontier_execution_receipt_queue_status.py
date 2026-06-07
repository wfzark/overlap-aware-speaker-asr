from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_status import build_status_row


class FrontierExecutionReceiptQueueStatusTest(unittest.TestCase):
    def test_build_status_row_marks_combined_ready_when_all_ready(self) -> None:
        row = build_status_row(
            {
                "meeteval_readiness_status": {"readiness_status": "receipt_ready_to_fill"},
                "speaker_profile_readiness_status": {"readiness_status": "receipt_ready_to_fill"},
                "external_staging_readiness_status": {"readiness_status": "receipt_ready_to_fill"},
            }
        )

        self.assertEqual(row["combined_readiness_status"], "receipt_ready_to_fill")

    def test_build_status_row_marks_combined_not_ready_when_one_pending(self) -> None:
        row = build_status_row(
            {
                "meeteval_readiness_status": {"readiness_status": "receipt_ready_to_fill"},
                "speaker_profile_readiness_status": {"readiness_status": "receipt_not_ready"},
                "external_staging_readiness_status": {"readiness_status": "receipt_ready_to_fill"},
            }
        )

        self.assertEqual(row["combined_readiness_status"], "receipt_not_ready")


if __name__ == "__main__":
    unittest.main()
