from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_handoff_readiness_bridge_checklist import build_bridge_checklist_rows


class SpeakerProfileEmbeddingTrialHandoffReadinessBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_embedding_trial(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "readiness_status": "handoff_ready",
                "trial_case_target": "NoOverlap",
            }
        )

        self.assertEqual(rows[0]["readiness_status"], "handoff_ready")
        self.assertIn("embedding_trial", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
