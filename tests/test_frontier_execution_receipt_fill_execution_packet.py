from __future__ import annotations

import unittest

from src.frontier_execution_receipt_fill_execution_packet import build_packet_rows


class FrontierExecutionReceiptFillExecutionPacketTest(unittest.TestCase):
    def test_build_packet_rows_includes_four_sections(self) -> None:
        rows = build_packet_rows({"combined_fill_status": "fill_queue_ready", "awaiting_fill_count": "3"})

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["section_name"], "fill_queue_status")


if __name__ == "__main__":
    unittest.main()
