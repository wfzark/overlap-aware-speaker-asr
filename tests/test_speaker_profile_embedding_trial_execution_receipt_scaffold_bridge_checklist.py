from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_scaffold_bridge_checklist import build_bridge_checklist_rows


class SpeakerProfileEmbeddingTrialExecutionReceiptScaffoldBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_execution_receipt(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "NoOverlap",
                "scaffold_status": "receipt_scaffold_only",
                "preflight_pass": "True",
                "swapped_bias_detected": "True",
            }
        )

        self.assertEqual(rows[0]["case_id"], "NoOverlap")
        self.assertIn("execution_receipt", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
