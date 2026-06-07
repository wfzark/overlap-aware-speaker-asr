from __future__ import annotations

import unittest

from src.llm_critic_review_pass_next import build_next_row, select_queue_row_for_case


class LlmCriticReviewPassNextTest(unittest.TestCase):
    def test_select_queue_row_for_case_finds_mid_overlap(self) -> None:
        row = select_queue_row_for_case(
            [
                {"case_id": "HeavyOverlap", "queue_order": "1"},
                {"case_id": "MidOverlap", "queue_order": "3"},
            ],
            "MidOverlap",
        )

        self.assertEqual(row["queue_order"], "3")

    def test_build_next_row_records_third_pass(self) -> None:
        row = build_next_row(
            {"case_id": "MidOverlap", "queue_order": "3"},
            {
                "case_id": "MidOverlap",
                "review_priority": "high",
                "review_outcome": "Qualitative critic pass recorded for MidOverlap; no verified transcript repair was applied.",
            },
            2,
        )

        self.assertEqual(row["completed_pass_count"], "2")
        self.assertIn("Third qualitative pass", row["next_note"])


if __name__ == "__main__":
    unittest.main()
