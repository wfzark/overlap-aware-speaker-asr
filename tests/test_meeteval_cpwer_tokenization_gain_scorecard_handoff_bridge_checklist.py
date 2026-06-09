from __future__ import annotations

import unittest

from src.meeteval_cpwer_tokenization_gain_scorecard_handoff_bridge_checklist import (
    build_bridge_checklist_rows,
)


class MeetEvalCpwerTokenizationGainScorecardHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_when_handoff_ready(self) -> None:
        handoff = {
            "handoff_status": "tokenization_gain_handoff_ready",
            "recommended_default_mode": "character_spaced",
            "adapted_and_aligned_count": "5",
            "case_count": "5",
        }

        rows = build_bridge_checklist_rows(handoff)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["handoff_status"], "tokenization_gain_handoff_ready")

    def test_build_bridge_checklist_rows_empty_when_handoff_missing(self) -> None:
        self.assertEqual(build_bridge_checklist_rows({}), [])


if __name__ == "__main__":
    unittest.main()
