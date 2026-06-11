from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_tokenization_gain_scorecard import (
    SCORECARD_COLUMNS,
    SUMMARY_COLUMNS,
    build_scorecard_rows,
    build_summary_row,
    write_outputs,
)


class MeetEvalCpwerTokenizationGainScorecardWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_scorecard_and_summary_artifacts(self) -> None:
        raw_rows = [{"case_id": "NoOverlap", "official_cpwer": "1.5"}]
        char_rows = [{"case_id": "NoOverlap", "official_cpwer": "0.12"}]
        rows = build_scorecard_rows(raw_rows, char_rows, {"NoOverlap": "0.12"})
        summary_row = build_summary_row(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.meeteval_cpwer_tokenization_gain_scorecard.PROJECT_ROOT", root):
                outputs = write_outputs(rows, summary_row)

            for path in outputs:
                self.assertTrue(path.exists())
            with outputs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCORECARD_COLUMNS)
                self.assertEqual(list(reader)[0]["adaptation_status"], "adapted_and_aligned")
            with outputs[2].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SUMMARY_COLUMNS)
            self.assertIn("Tokenization Gain Scorecard", outputs[4].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
