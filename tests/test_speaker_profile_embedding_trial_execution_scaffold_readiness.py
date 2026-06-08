from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_scaffold_readiness import build_readiness_row


class SpeakerProfileEmbeddingTrialExecutionScaffoldReadinessTest(unittest.TestCase):
    def test_build_readiness_row_marks_scaffold_ready(self) -> None:
        row = build_readiness_row(
            {"queue_status": "queue_complete", "trial_case_target": "NoOverlap"},
            {"scaffold_status": "execution_scaffold_only", "case_id": "NoOverlap"},
        )

        self.assertEqual(row["readiness_status"], "scaffold_ready")
        self.assertEqual(row["case_id"], "NoOverlap")


if __name__ == "__main__":
    unittest.main()
