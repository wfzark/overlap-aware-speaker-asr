from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_readiness_bridge_checklist import (
    build_bridge_checklist_rows,
)


class SpeakerProfileEmbeddingTrialExecutionReceiptReadinessBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_records_swapped_bias(self) -> None:
        rows = build_bridge_checklist_rows(
            {"case_id": "NoOverlap", "readiness_status": "receipt_ready_to_fill", "swapped_bias_detected": "True"}
        )

        self.assertIn("swapped_bias_detected=True", rows[0]["bridge_note"])


if __name__ == "__main__":
    unittest.main()
