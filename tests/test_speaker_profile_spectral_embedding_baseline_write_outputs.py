from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.speaker_profile_spectral_embedding_baseline import BASELINE_COLUMNS, write_outputs


def _sample_row() -> dict[str, str]:
    return {
        "case_id": "NoOverlap",
        "trial_scope": "NoOverlap_only",
        "trial_status": "executed_baseline",
        "text_best_alignment": "swapped",
        "spectral_best_alignment": "swapped",
        "signals_agree": "True",
        "text_confidence_gap": "0.42",
        "spectral_confidence_gap": "0.01",
        "direct_text_score": "0.39",
        "swapped_text_score": "0.81",
        "direct_spectral_score": "0.50",
        "swapped_spectral_score": "0.51",
        "result_label": "experimental/frontier",
        "observation": "fixture",
    }


class SpeakerProfileSpectralEmbeddingBaselineWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_csv_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.speaker_profile_spectral_embedding_baseline.PROJECT_ROOT", root):
                csv_path, json_path, md_path = write_outputs(_sample_row())

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists(), msg=str(path))

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, BASELINE_COLUMNS)
                self.assertEqual(list(reader)[0]["case_id"], "NoOverlap")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["trial_status"], "executed_baseline")
            self.assertIn("experimental/frontier", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
