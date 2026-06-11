from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.speaker_profile_embedding_trial import (
    TRIAL_COLUMNS,
    build_trial_receipt_rows,
    build_trial_row,
    write_outputs,
)


class SpeakerProfileEmbeddingTrialWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_trial_and_receipt_artifacts(self) -> None:
        trial_row = build_trial_row(
            {"trial_case_target": "NoOverlap", "method_direction": "embedding_or_voiceprint_baseline"},
            {
                "case_id": "NoOverlap",
                "direct_profile_score": "0.42",
                "swapped_profile_score": "0.18",
                "profile_confidence_gap": "0.24",
            },
        )
        receipt_rows = build_trial_receipt_rows(trial_row)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.speaker_profile_embedding_trial.PROJECT_ROOT", root):
                paths = write_outputs(trial_row, receipt_rows)

            for path in paths:
                self.assertTrue(path.exists())
            with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, TRIAL_COLUMNS)
                self.assertEqual(list(reader)[0]["case_id"], "NoOverlap")
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(payload["trial_status"], "scaffold_only")
            self.assertIn("Embedding Trial", paths[2].read_text(encoding="utf-8"))
            receipt_payload = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(receipt_payload[0]["execution_status"], "trial_scaffold_complete")


if __name__ == "__main__":
    unittest.main()
