from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_writeback_status import (
    build_status_summary,
    derive_writeback_status,
)


class FrontierExecutionReceiptQueueWritebackStatusTest(unittest.TestCase):
    def test_derive_writeback_status_awaiting_for_template(self) -> None:
        self.assertEqual(
            derive_writeback_status("template_only", "receipt_ready_to_fill"),
            "awaiting_writeback",
        )

    def test_build_status_summary_reports_in_progress_for_mixed_rows(self) -> None:
        summary = build_status_summary(
            [
                {"writeback_status": "writeback_complete"},
                {"writeback_status": "awaiting_writeback"},
                {"writeback_status": "awaiting_writeback"},
            ]
        )

        self.assertEqual(summary["combined_writeback_status"], "writeback_queue_in_progress")
        self.assertEqual(summary["writeback_complete_count"], "1")


if __name__ == "__main__":
    unittest.main()
