from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_status_bridge_checklist import build_bridge_checklist_rows


class SpeakerProfileEmbeddingTrialExecutionStatusBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_records_swapped_bias(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "NoOverlap",
                "execution_chain_status": "execution_chain_ready",
                "swapped_bias_detected": "True",
                "preflight_pass": "True",
            }
        )

        self.assertEqual(rows[0]["swapped_bias_detected"], "True")
        self.assertIn("swapped_bias_detected=True", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_defaults_case_id(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows[0]["case_id"], "NoOverlap")


if __name__ == "__main__":
    unittest.main()
