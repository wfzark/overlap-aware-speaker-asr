from __future__ import annotations

import unittest

from src.meeteval_cpwer_tokenization_gain_scorecard_handoff import build_handoff_row


class MeetEvalCpwerTokenizationGainScorecardHandoffTest(unittest.TestCase):
    def test_build_handoff_row_when_character_spaced_default(self) -> None:
        summary = {
            "recommended_default_mode": "character_spaced",
            "adapted_and_aligned_count": "5",
            "case_count": "5",
        }

        row = build_handoff_row(summary)

        self.assertEqual(row["handoff_status"], "tokenization_gain_handoff_ready")
        self.assertIn("tokenization_adaptation_completion_summary", row["handoff_target"])

    def test_build_handoff_row_pending_when_review_required(self) -> None:
        summary = {
            "recommended_default_mode": "case_by_case_review",
            "adapted_and_aligned_count": "3",
            "case_count": "5",
        }

        row = build_handoff_row(summary)

        self.assertEqual(row["handoff_status"], "tokenization_gain_handoff_pending")


if __name__ == "__main__":
    unittest.main()
