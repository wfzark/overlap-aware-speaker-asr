from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_alignment_drift_segment_inspection import (
    INSPECTION_COLUMNS,
    build_inspection_receipt_row,
    write_outputs,
)


class MeetEvalCpwerAlignmentDriftSegmentInspectionWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_inspection_and_receipt_artifacts(self) -> None:
        inspection = {
            "case_id": "HeavyOverlap",
            "hypothesis_source": "separated_whisper_cleaned",
            "reference_segment_count": 4,
            "hypothesis_segment_count": 4,
            "segment_count_delta": 0,
            "speaker_set_match": True,
            "time_range_valid": True,
            "export_path_valid": True,
            "inspection_pass": True,
            "inspection_note": "fixture",
        }
        receipt_row = build_inspection_receipt_row(inspection)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.meeteval_cpwer_alignment_drift_segment_inspection.PROJECT_ROOT", root):
                outputs = write_outputs(inspection, receipt_row)

            for path in outputs:
                self.assertTrue(path.exists())
            with outputs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, INSPECTION_COLUMNS)
                self.assertEqual(list(reader)[0]["case_id"], "HeavyOverlap")
            self.assertIn("MeetEval cpWER Alignment Drift Segment Inspection", outputs[2].read_text(encoding="utf-8"))
            self.assertEqual(json.loads(outputs[3].read_text(encoding="utf-8"))[0]["execution_status"], "segment_inspection_complete")


if __name__ == "__main__":
    unittest.main()
