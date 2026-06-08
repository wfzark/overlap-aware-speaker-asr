from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_completion_dashboard import build_dashboard_row


class FrontierExecutionReceiptQueueCompletionDashboardTest(unittest.TestCase):
    def test_build_dashboard_row_summarizes_receipt_queue_state(self) -> None:
        row = build_dashboard_row(
            {"operator_frontier": "meeteval_compatibility"},
            {
                "next_milestone": "first_receipt_queue_checkpoint_complete",
                "remaining_frontier_count": "2",
            },
        )

        self.assertEqual(row["current_first_frontier"], "meeteval_compatibility")
        self.assertEqual(row["next_milestone"], "first_receipt_queue_checkpoint_complete")
        self.assertEqual(row["remaining_frontier_count"], "2")
        self.assertEqual(row["dominant_blocker"], "receipt_template_fill_pending")

    def test_build_dashboard_row_returns_empty_without_inputs(self) -> None:
        row = build_dashboard_row({}, {})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
