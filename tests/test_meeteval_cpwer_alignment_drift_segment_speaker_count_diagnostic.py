from __future__ import annotations

import unittest
from collections import Counter

from src.meeteval_cpwer_alignment_drift_segment_speaker_count_diagnostic import (
    build_receipt_row,
    build_speaker_rows,
    build_summary_row,
    run_speaker_count_diagnostic,
)


class MeetEvalCpwerAlignmentDriftSegmentSpeakerCountDiagnosticTest(unittest.TestCase):
    def test_build_speaker_rows_report_delta(self) -> None:
        rows = build_speaker_rows(
            "HeavyOverlap",
            Counter({"SPEAKER_1": 10, "SPEAKER_2": 15}),
            Counter({"SPEAKER_1": 12, "SPEAKER_2": 13}),
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["count_match"], "False")

    def test_build_summary_row_counts_mismatches(self) -> None:
        summary = build_summary_row(
            "HeavyOverlap",
            [
                {
                    "case_id": "HeavyOverlap",
                    "speaker": "SPEAKER_1",
                    "reference_segment_count": "10",
                    "hypothesis_segment_count": "12",
                    "segment_count_delta": "2",
                    "count_match": "False",
                }
            ],
        )

        self.assertEqual(summary["mismatched_speaker_count"], "1")

    def test_run_speaker_count_diagnostic_reports_heavy_overlap(self) -> None:
        speaker_rows, summary = run_speaker_count_diagnostic("HeavyOverlap")

        self.assertTrue(speaker_rows)
        self.assertEqual(summary["case_id"], "HeavyOverlap")

    def test_build_receipt_row_marks_complete(self) -> None:
        row = build_receipt_row({"case_id": "HeavyOverlap", "mismatched_speaker_count": "2"})

        self.assertEqual(row["execution_status"], "speaker_count_diagnostic_complete")


if __name__ == "__main__":
    unittest.main()
