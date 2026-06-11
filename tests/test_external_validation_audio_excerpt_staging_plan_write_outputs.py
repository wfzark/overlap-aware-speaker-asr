from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.external_validation_audio_excerpt_staging_plan import PLAN_COLUMNS, write_outputs


class ExternalValidationAudioExcerptStagingPlanWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_artifacts(self) -> None:
        row = {
            "dataset_name": "AISHELL-4",
            "slice_id": "stub",
            "label": "external/sanity-check",
            "license_status": "confirmed_research_cc_by_sa_4_0",
            "audio_path": "resources/external_sanity_check/aishell4/a.wav",
            "reference_path": "resources/external_sanity_check/aishell4/a.json",
            "audio_staged": "False",
            "reference_template_staged": "True",
            "staging_status": "awaiting_local_audio_download",
            "download_source": "https://www.openslr.org/111/",
            "result_label": "external/sanity-check",
            "staging_note": "plan only",
        }
        summary = [{"metric": "audio_staged", "value": "False", "label": "external/sanity-check"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.external_validation_audio_excerpt_staging_plan.PROJECT_ROOT", root):
                paths = write_outputs(row, summary)
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
            with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, PLAN_COLUMNS)


if __name__ == "__main__":
    unittest.main()
