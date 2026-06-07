from __future__ import annotations

import unittest

from src.speaker_profile_text_proxy_trial_diagnostic_completion_summary import build_completion_row


class SpeakerProfileTextProxyTrialDiagnosticCompletionSummaryTest(unittest.TestCase):
    def test_build_completion_row_marks_queue_complete_when_all_swapped(self) -> None:
        row = build_completion_row(
            {
                "swapped_count": "5",
                "case_count": "5",
                "average_confidence_gap": "0.12",
            }
        )

        self.assertEqual(row["queue_status"], "queue_complete")
        self.assertEqual(row["swapped_count"], "5")


if __name__ == "__main__":
    unittest.main()
