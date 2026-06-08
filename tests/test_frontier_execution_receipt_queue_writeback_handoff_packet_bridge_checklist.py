from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_writeback_handoff_packet_bridge_checklist import (
    build_bridge_checklist_rows,
)


class FrontierExecutionReceiptQueueWritebackHandoffPacketBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_writeback_status(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "combined_writeback_status": "writeback_queue_in_progress",
                "awaiting_writeback_count": "2",
                "writeback_complete_count": "1",
            }
        )

        self.assertEqual(rows[0]["combined_writeback_status"], "writeback_queue_in_progress")
        self.assertIn("writeback_status", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
