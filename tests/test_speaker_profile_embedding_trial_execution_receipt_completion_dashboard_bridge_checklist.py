from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_completion_dashboard_bridge_checklist import (
    build_bridge_checklist_rows,
)


class SpeakerProfileEmbeddingTrialExecutionReceiptCompletionDashboardBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_readiness(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "current_case": "NoOverlap",
                "dashboard_note": "NoOverlap remains in receipt_ready_to_fill.",
            }
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["current_case"], "NoOverlap")
        self.assertIn("speaker_profile_embedding_trial_execution_receipt_readiness.md", rows[0]["receipt_target"])
        self.assertIn("NoOverlap remains in receipt_ready_to_fill", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_returns_empty_without_dashboard(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
