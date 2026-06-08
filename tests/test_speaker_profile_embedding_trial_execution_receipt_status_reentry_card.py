from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_status_reentry_card import build_reentry_card_row


class SpeakerProfileEmbeddingTrialExecutionReceiptStatusReentryCardTest(unittest.TestCase):
    def test_build_reentry_card_row_uses_preflight_and_status(self) -> None:
        row = build_reentry_card_row(
            [
                {
                    "current_case": "NoOverlap",
                    "receipt_target": "results/figures/speaker_profile_embedding_trial_execution_status.md",
                }
            ],
            {"execution_chain_status": "execution_chain_ready"},
        )

        self.assertEqual(row["current_case"], "NoOverlap")
        self.assertEqual(
            row["status_rollup_target"],
            "results/figures/speaker_profile_embedding_trial_execution_status.md",
        )
        self.assertEqual(row["execution_chain_status"], "execution_chain_ready")
        self.assertIn("refresh the speaker-profile status rollup", row["reentry_action"])

    def test_build_reentry_card_row_returns_empty_without_inputs(self) -> None:
        row = build_reentry_card_row([], {})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
