from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_open_card import (
    build_receipt_open_card_row,
)


class SpeakerProfileEmbeddingTrialExecutionReceiptOpenCardTest(unittest.TestCase):
    def test_build_receipt_open_card_row_targets_receipt(self) -> None:
        row = build_receipt_open_card_row(
            [
                {
                    "case_id": "NoOverlap",
                    "readiness_status": "receipt_ready_to_fill",
                    "receipt_target": "results/tables/speaker_profile_embedding_trial_execution_receipt.json",
                }
            ]
        )

        self.assertEqual(row["case_id"], "NoOverlap")
        self.assertEqual(row["readiness_status"], "receipt_ready_to_fill")
        self.assertIn("speaker_profile_embedding_trial_execution_receipt.json", row["receipt_target"])
        self.assertIn("Open", row["open_action"])

    def test_build_receipt_open_card_row_returns_empty_without_bridge_rows(self) -> None:
        row = build_receipt_open_card_row([])

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
