from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.meeteval_dry_run import build_diagnostic_receipt_row, write_outputs


class MeetEvalDryRunWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_writes_diagnostic_and_receipt_artifacts(self) -> None:
        diagnostic = {
            "case_id": "NoOverlap",
            "hypothesis_source": "separated_whisper",
            "reference_segment_count": 2,
            "hypothesis_segment_count": 2,
            "speaker_set_match": "yes",
            "time_range_valid": "yes",
            "export_path_valid": "yes",
            "diagnostic_pass": "pass",
            "diagnostic_note": "experimental/frontier dry-run diagnostic only",
        }
        receipt_row = build_diagnostic_receipt_row(diagnostic)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with unittest.mock.patch("src.meeteval_dry_run.PROJECT_ROOT", root):
                paths = write_outputs(diagnostic, receipt_row)
                for path in paths:
                    self.assertTrue(path.exists())
                diagnostic_json = json.loads(paths[1].read_text(encoding="utf-8"))
                diagnostic_md = paths[2].read_text(encoding="utf-8")
                receipt_json = json.loads(paths[3].read_text(encoding="utf-8"))
        self.assertEqual(diagnostic_json["case_id"], "NoOverlap")
        self.assertIn("experimental/frontier", diagnostic_md)
        self.assertEqual(receipt_json[0]["execution_status"], "diagnostic_complete")


if __name__ == "__main__":
    unittest.main()
