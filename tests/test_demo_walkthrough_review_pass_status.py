from __future__ import annotations

import unittest

from src.demo_walkthrough_review_pass_status import build_status_row


class DemoWalkthroughReviewPassStatusTest(unittest.TestCase):
    def test_build_status_row_marks_queue_complete(self) -> None:
        row = build_status_row(
            [{"step_id": "1"}, {"step_id": "2"}, {"step_id": "3"}, {"step_id": "4"}, {"step_id": "5"}],
            {"1", "2", "3", "4", "5"},
        )

        self.assertEqual(row["queue_status"], "queue_complete")
        self.assertEqual(row["pending_count"], "0")

    def test_build_status_row_marks_in_progress(self) -> None:
        row = build_status_row(
            [{"step_id": "1"}, {"step_id": "2"}, {"step_id": "3"}],
            {"1"},
        )

        self.assertEqual(row["queue_status"], "queue_in_progress")
        self.assertEqual(row["pending_count"], "2")


if __name__ == "__main__":
    unittest.main()
