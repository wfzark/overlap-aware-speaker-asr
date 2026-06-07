from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_completion_summary_bridge_checklist import build_bridge_checklist_rows


class FrontierExecutionReceiptQueueCompletionSummaryBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_uses_queue_complete(self) -> None:
        rows = build_bridge_checklist_rows({"queue_status": "queue_complete", "ready_receipt_count": "3"})

        self.assertEqual(rows[0]["queue_status"], "queue_complete")


if __name__ == "__main__":
    unittest.main()
