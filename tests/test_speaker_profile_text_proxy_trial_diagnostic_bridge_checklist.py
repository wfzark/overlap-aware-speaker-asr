from __future__ import annotations

import unittest

from src.speaker_profile_text_proxy_trial_diagnostic_bridge_checklist import build_bridge_checklist_rows


class SpeakerProfileTextProxyTrialDiagnosticBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_embedding_handoff(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "swapped_count": "5",
                "case_count": "5",
                "next_method_direction": "embedding_or_voiceprint_baseline",
            }
        )

        self.assertEqual(rows[0]["swapped_count"], "5")
        self.assertIn("embedding_trial_handoff", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
