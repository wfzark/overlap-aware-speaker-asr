from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_preflight_readiness import build_readiness_row


class SpeakerProfilePreflightReadinessTest(unittest.TestCase):
    def test_build_readiness_row_marks_preflight_ready(self) -> None:
        row = build_readiness_row(
            {"queue_status": "queue_complete", "case_id": "NoOverlap"},
            {"preflight_pass": True, "case_id": "NoOverlap"},
        )

        self.assertEqual(row["readiness_status"], "preflight_ready")


if __name__ == "__main__":
    unittest.main()
