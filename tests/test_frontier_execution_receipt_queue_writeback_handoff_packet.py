from __future__ import annotations

import unittest

from src.frontier_execution_receipt_queue_writeback_handoff_packet import build_packet_rows


class FrontierExecutionReceiptQueueWritebackHandoffPacketTest(unittest.TestCase):
    def test_build_packet_rows_include_five_sections(self) -> None:
        rows = build_packet_rows(
            {
                "combined_writeback_status": "writeback_queue_in_progress",
                "awaiting_writeback_count": "2",
                "writeback_complete_count": "1",
            }
        )

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["section_name"], "receipt_queue_writeback_status")
        self.assertEqual(rows[-1]["section_name"], "receipt_queue_writeback_open_card_bridge_checklist")


if __name__ == "__main__":
    unittest.main()
