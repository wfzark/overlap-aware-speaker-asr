from __future__ import annotations

import unittest

from src.demo_excellence_queue_status import build_status_row


class DemoExcellenceQueueStatusTest(unittest.TestCase):
    def test_build_status_row_combined_complete_when_both_complete(self) -> None:
        row = build_status_row(
            {"queue_status": "queue_complete"},
            {"queue_status": "queue_complete"},
        )

        self.assertEqual(row["combined_queue_status"], "queue_complete")

    def test_build_status_row_combined_in_progress_when_one_incomplete(self) -> None:
        row = build_status_row(
            {"queue_status": "queue_complete"},
            {"queue_status": "queue_in_progress"},
        )

        self.assertEqual(row["combined_queue_status"], "queue_in_progress")


if __name__ == "__main__":
    unittest.main()
