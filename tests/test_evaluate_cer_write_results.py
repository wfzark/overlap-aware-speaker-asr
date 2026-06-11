from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate_cer import CSV_COLUMNS, write_cer_results


def _sample_cer_row() -> dict[str, object]:
    return {
        "case_id": "FixtureCase",
        "method": "mixed_whisper",
        "reference_type": "verified_reference",
        "hypothesis_path": "results/transcripts_raw/FixtureCase_mixed_whisper.json",
        "reference_length": 10,
        "hypothesis_length": 11,
        "edit_distance": 1,
        "cer": 0.1,
    }


class EvaluateCerWriteResultsTest(unittest.TestCase):
    def test_write_cer_results_emits_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "tables"
            csv_path, json_path = write_cer_results([_sample_cer_row()], output_dir=output_dir)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, CSV_COLUMNS)
                rows = list(reader)
            self.assertEqual(rows[0]["case_id"], "FixtureCase")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["cer"], 0.1)


if __name__ == "__main__":
    unittest.main()
