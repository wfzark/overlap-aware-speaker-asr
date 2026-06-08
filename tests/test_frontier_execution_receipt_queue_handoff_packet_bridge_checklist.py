from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_handoff_packet_bridge_checklist import (
    build_bridge_checklist_rows,
)


class FrontierExecutionReceiptQueueHandoffPacketBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_operator_brief(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "queue_status": "queue_complete",
                "ready_receipt_count": "3",
                "pending_receipt_count": "0",
            }
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertIn("frontier_execution_receipt_queue_operator_brief.md", rows[0]["receipt_target"])
        self.assertIn("ready_receipt_count=3", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_defaults_queue_status(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows[0]["queue_status"], "queue_in_progress")


if __name__ == "__main__":
    unittest.main()
