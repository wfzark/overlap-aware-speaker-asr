from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_alignment_drift_segment_redistribution_diagnostic import (
    SPEAKER_COLUMNS,
    SUMMARY_COLUMNS,
    build_receipt_row,
    build_speaker_rows,
    build_summary_row,
    write_outputs,
)


class MeetEvalCpwerAlignmentDriftSegmentRedistributionDiagnosticWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_speaker_summary_and_receipt_artifacts(self) -> None:
        granularity_rows = [
            {
                "case_id": "HeavyOverlap",
                "speaker": "SPEAKER_1",
                "reference_segment_count": "2",
                "hypothesis_segment_count": "3",
            },
            {
                "case_id": "HeavyOverlap",
                "speaker": "SPEAKER_2",
                "reference_segment_count": "2",
                "hypothesis_segment_count": "2",
            },
        ]
        speaker_rows = build_speaker_rows("HeavyOverlap", granularity_rows)
        summary = build_summary_row("HeavyOverlap", speaker_rows)
        receipt_row = build_receipt_row(summary)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.meeteval_cpwer_alignment_drift_segment_redistribution_diagnostic.PROJECT_ROOT",
                root,
            ):
                outputs = write_outputs(speaker_rows, summary, receipt_row)

            for path in outputs:
                self.assertTrue(path.exists())
            with outputs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SPEAKER_COLUMNS)
                self.assertEqual(list(reader)[0]["redistribution_pattern"], "hypothesis_split")
            with outputs[3].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SUMMARY_COLUMNS)
            self.assertEqual(
                json.loads(outputs[6].read_text(encoding="utf-8"))[0]["execution_status"],
                "redistribution_diagnostic_complete",
            )


if __name__ == "__main__":
    unittest.main()
