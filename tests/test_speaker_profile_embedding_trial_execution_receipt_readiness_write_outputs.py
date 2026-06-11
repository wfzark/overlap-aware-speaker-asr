from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.speaker_profile_embedding_trial_execution_receipt_readiness import (
    READINESS_COLUMNS,
    build_readiness_row,
    write_outputs,
)


class SpeakerProfileEmbeddingTrialExecutionReceiptReadinessWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_readiness_artifacts(self) -> None:
        readiness_row = build_readiness_row(
            {
                "case_id": "NoOverlap",
                "execution_chain_status": "execution_chain_ready",
                "preflight_pass": "True",
            },
            {
                "execution_status": "template_only",
                "preflight_pass": "True",
                "swapped_bias_detected": "True",
                "combined_signal_status": "mixed_signal",
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.speaker_profile_embedding_trial_execution_receipt_readiness.PROJECT_ROOT",
                root,
            ):
                csv_path, json_path, md_path = write_outputs(readiness_row)

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, READINESS_COLUMNS)
                self.assertEqual(list(reader)[0]["readiness_status"], "receipt_ready_to_fill")
            self.assertIn("Receipt Readiness", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
