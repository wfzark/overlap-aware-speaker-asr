from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_runbook_card import build_runbook_card_row


class SpeakerProfileEmbeddingTrialExecutionReceiptRunbookCardTest(unittest.TestCase):
    def test_build_runbook_card_row_targets_current_case(self) -> None:
        row = build_runbook_card_row(
            {
                "operator_case": "NoOverlap",
                "operator_status": "receipt_ready_to_fill",
                "operator_target": "results/figures/speaker_profile_embedding_trial_execution_receipt_readiness.md",
                "operator_action": "Reopen readiness target for NoOverlap.",
                "operator_evidence": "bridge artifacts",
            },
            {
                "checklist_order": "1",
                "next_gate": "Confirm this bridge before opening results/figures/speaker_profile_embedding_trial_execution_receipt_readiness.md.",
            },
        )

        self.assertEqual(row["recommended_case"], "NoOverlap")
        self.assertEqual(row["readiness_status"], "receipt_ready_to_fill")
        self.assertIn("receipt_readiness.md", row["receipt_target"])
        self.assertIn("checklist_order=1", row["runbook_note"])

    def test_build_runbook_card_row_returns_empty_without_operator_brief(self) -> None:
        row = build_runbook_card_row({}, {})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
