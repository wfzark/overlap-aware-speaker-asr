from __future__ import annotations

import unittest

from src.llm_critic_review_pass_status import build_status_rows, build_summary_row


class LlmCriticReviewPassStatusTest(unittest.TestCase):
    def test_build_status_rows_mark_completed_cases(self) -> None:
        rows = build_status_rows(
            [
                {"queue_order": "1", "case_id": "HeavyOverlap", "review_priority": "high"},
                {"queue_order": "2", "case_id": "LightOverlap", "review_priority": "high"},
            ],
            {"HeavyOverlap", "LightOverlap"},
        )

        self.assertEqual(rows[0]["pass_status"], "review_complete")
        self.assertEqual(rows[1]["pass_status"], "review_complete")

    def test_build_summary_row_points_to_next_pending_case(self) -> None:
        summary = build_summary_row(
            [
                {"pass_status": "review_complete", "case_id": "HeavyOverlap"},
                {"pass_status": "pending_review", "case_id": "MidOverlap"},
            ]
        )

        self.assertEqual(summary["completed_count"], "1")
        self.assertEqual(summary["next_case_id"], "MidOverlap")


if __name__ == "__main__":
    unittest.main()
