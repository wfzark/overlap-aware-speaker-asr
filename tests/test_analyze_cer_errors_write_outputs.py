from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.analyze_cer_errors import write_outputs


class AnalyzeCerErrorsWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_writes_markdown_and_json(self) -> None:
        report = {
            "case_id": "FixtureCase",
            "method": "mixed_whisper",
            "reference_length": 10,
            "hypothesis_length": 12,
            "edit_distance": 2,
            "cer": 0.2,
            "suspected_repeated_phrases": [{"type": "repeated_clause", "phrase": "重复", "count": 2}],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "error_analysis"
            with unittest.mock.patch("src.analyze_cer_errors.PROJECT_ROOT", Path(tmp_dir)):
                md_path, json_path = write_outputs(report)
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            md_text = md_path.read_text(encoding="utf-8-sig")
            self.assertIn("FixtureCase", md_text)
            self.assertIn("Suspected Repeated Phrases", md_text)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["case_id"], "FixtureCase")
            self.assertEqual(len(payload["suspected_repeated_phrases"]), 1)


if __name__ == "__main__":
    unittest.main()
