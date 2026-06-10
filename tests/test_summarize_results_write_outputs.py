from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.summarize_results import SUMMARY_COLUMNS, write_outputs


def _sample_summary_row() -> dict[str, object]:
    return {
        "case_id": "FixtureCase",
        "overlap_level": 0,
        "mixed_cer": 0.2,
        "separated_cer": 0.15,
        "separated_cleaned_cer": 0.14,
        "best_method": "separated_whisper_cleaned",
        "observation": "Fixture observation",
    }


class SummarizeResultsWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_csv_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.summarize_results.PROJECT_ROOT", root):
                csv_path, json_path, md_path = write_outputs([_sample_summary_row()])

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SUMMARY_COLUMNS)
                rows = list(reader)
            self.assertEqual(rows[0]["case_id"], "FixtureCase")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["best_method"], "separated_whisper_cleaned")

            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("| case_id |", markdown)
            self.assertIn("FixtureCase", markdown)


if __name__ == "__main__":
    unittest.main()
