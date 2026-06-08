from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_status_reentry_bridge_checklist import (
    build_bridge_checklist_rows,
)


class SpeakerProfileEmbeddingTrialExecutionReceiptStatusReentryBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_links_reentry_to_readiness_reopen(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "current_case": "NoOverlap",
                "reentry_action": (
                    "After status preflight is confirmed, reopen "
                    "results/figures/speaker_profile_embedding_trial_execution_status.md "
                    "and refresh the speaker-profile status rollup."
                ),
            }
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["current_case"], "NoOverlap")
        self.assertIn(
            "speaker_profile_embedding_trial_execution_receipt_readiness.md",
            rows[0]["receipt_target"],
        )
        self.assertIn("refresh the speaker-profile status rollup", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_returns_empty_without_reentry_card(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
