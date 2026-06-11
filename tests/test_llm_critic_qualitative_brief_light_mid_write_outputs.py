from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_critic_qualitative_brief_light_mid import BRIEF_COLUMNS, write_outputs


class LlmCriticQualitativeBriefLightMidWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_artifacts(self) -> None:
        row = {
            "case_id": "MidOverlap",
            "overlap_ratio_anchor": "0.375",
            "mixed_cer": "0.18",
            "separated_cer": "0.27",
            "separated_cleaned_cer": "0.21",
            "dominant_error_mixed": "deletion",
            "dominant_error_separated": "insertion",
            "separation_harm_observed": "True",
            "critic_hypothesis": "test",
            "candidate_repair": "prefer mixed",
            "uncertainty_note": "qualitative only",
            "result_label": "qualitative/demo",
        }
        summary = [{"metric": "target_case_count", "value": "1", "label": "stable/gold"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.llm_critic_qualitative_brief_light_mid.PROJECT_ROOT", root):
                paths = write_outputs([row], summary)
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
            with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, BRIEF_COLUMNS)


if __name__ == "__main__":
    unittest.main()
