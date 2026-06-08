from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge_checklist import (
    build_bridge_checklist_rows,
)


class SpeakerProfileEmbeddingTrialExecutionReceiptOperatorBriefBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_readiness_gate(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "operator_case": "NoOverlap",
                "prerequisite_artifact": "results/figures/speaker_profile_embedding_trial_execution_receipt_operator_brief.md",
                "receipt_target": "results/figures/speaker_profile_embedding_trial_execution_receipt_readiness.md",
                "bridge_note": "Open the operator brief first.",
            }
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operator_case"], "NoOverlap")
        self.assertIn("receipt_readiness.md", rows[0]["next_gate"])

    def test_build_bridge_checklist_rows_returns_empty_without_bridge(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
