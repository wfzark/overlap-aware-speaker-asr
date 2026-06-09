from __future__ import annotations

import unittest

from src.meeteval_cpwer_tokenization_gain_scorecard_bridge_checklist import build_bridge_checklist_rows


class MeetEvalCpwerTokenizationGainScorecardBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_when_character_spaced_default(self) -> None:
        summary = {
            "recommended_default_mode": "character_spaced",
            "adapted_and_aligned_count": "5",
            "case_count": "5",
            "average_raw_to_character_gain": "3.679091",
        }

        rows = build_bridge_checklist_rows(summary)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recommended_default_mode"], "character_spaced")
        self.assertIn("tokenization adaptation completion summary", rows[0]["next_gate"].lower())

    def test_build_bridge_checklist_rows_empty_when_summary_missing(self) -> None:
        self.assertEqual(build_bridge_checklist_rows({}), [])


if __name__ == "__main__":
    unittest.main()
