from __future__ import annotations

import unittest

from src.llm_critic_review_pass_continue import build_continue_row, select_queue_row_for_case


class LlmCriticReviewPassContinueTest(unittest.TestCase):
    def test_select_queue_row_for_case_finds_no_overlap(self) -> None:
        row = select_queue_row_for_case(
            [
                {"case_id": "MidOverlap", "queue_order": "3"},
                {"case_id": "NoOverlap", "queue_order": "4"},
            ],
            "NoOverlap",
        )

        self.assertEqual(row["queue_order"], "4")

    def test_build_continue_row_records_fourth_pass(self) -> None:
        row = build_continue_row(
            {"case_id": "NoOverlap", "queue_order": "4"},
            {
                "case_id": "NoOverlap",
                "review_priority": "high",
                "review_outcome": "Qualitative critic pass recorded for NoOverlap; no verified transcript repair was applied.",
            },
            3,
        )

        self.assertEqual(row["completed_pass_count"], "3")
        self.assertIn("Fourth qualitative pass", row["continue_note"])


if __name__ == "__main__":
    unittest.main()
