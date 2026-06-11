from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic import (
    DIAGNOSTIC_COLUMNS,
    build_diagnostic_receipt_row,
    write_outputs,
)


class MeetEvalCpwerAlignmentDriftSegmentReconciliationDiagnosticWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_diagnostic_and_receipt_artifacts(self) -> None:
        diagnostic = {
            "case_id": "HeavyOverlap",
            "hypothesis_source": "separated_whisper_cleaned",
            "reference_segment_count": 4,
            "hypothesis_segment_count": 4,
            "speaker_segment_count_match": True,
            "speaker_set_match": True,
            "time_range_valid": True,
            "export_path_valid": True,
            "reconciliation_pass": False,
            "diagnostic_note": "fixture",
        }
        receipt_row = build_diagnostic_receipt_row(diagnostic)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.meeteval_cpwer_alignment_drift_segment_reconciliation_diagnostic.PROJECT_ROOT",
                root,
            ):
                outputs = write_outputs(diagnostic, receipt_row)

            for path in outputs:
                self.assertTrue(path.exists())
            with outputs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, DIAGNOSTIC_COLUMNS)
                self.assertEqual(list(reader)[0]["case_id"], "HeavyOverlap")
            self.assertIn("Reconciliation Diagnostic", outputs[2].read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(outputs[3].read_text(encoding="utf-8"))[0]["execution_status"],
                "reconciliation_diagnostic_complete",
            )


if __name__ == "__main__":
    unittest.main()
