from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_official_execution_tokenization_diagnostic import (
    DIAGNOSTIC_COLUMNS,
    build_diagnostic_row,
    write_outputs,
)


class MeetEvalCpwerOfficialExecutionTokenizationDiagnosticWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_diagnostic_artifacts(self) -> None:
        segments = [
            {"speaker": "SPEAKER_1", "text": "你好世界"},
            {"speaker": "SPEAKER_2", "text": "测试文本"},
        ]
        with patch(
            "src.meeteval_cpwer_official_execution_tokenization_diagnostic.load_jsonl_segments",
            return_value=segments,
        ):
            row = build_diagnostic_row("NoOverlap", {"official_cpwer": "1.5"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.meeteval_cpwer_official_execution_tokenization_diagnostic.PROJECT_ROOT",
                root,
            ):
                csv_path, json_path, md_path = write_outputs([row])

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, DIAGNOSTIC_COLUMNS)
                self.assertEqual(list(reader)[0]["root_cause"], "no_whitespace_word_tokenization")
            self.assertIn("Tokenization Diagnostic", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
