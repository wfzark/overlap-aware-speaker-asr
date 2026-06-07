from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_inspection import (
    build_inspection_receipt_row,
    run_segment_inspection,
)


class MeetEvalCpwerAlignmentDriftSegmentInspectionTest(unittest.TestCase):
    def test_run_segment_inspection_reports_delta(self) -> None:
        inspection = run_segment_inspection("HeavyOverlap")

        self.assertEqual(inspection["case_id"], "HeavyOverlap")
        self.assertIn("segment_count_delta", inspection)

    def test_build_inspection_receipt_row_marks_complete(self) -> None:
        row = build_inspection_receipt_row(
            {
                "case_id": "HeavyOverlap",
                "hypothesis_source": "separated_whisper_cleaned",
                "reference_segment_count": 2,
                "hypothesis_segment_count": 2,
                "segment_count_delta": 0,
                "inspection_pass": True,
            }
        )

        self.assertEqual(row["execution_status"], "segment_inspection_complete")


if __name__ == "__main__":
    unittest.main()
