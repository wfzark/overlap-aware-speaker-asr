from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_timing_diagnostic import (
    build_receipt_row,
    build_speaker_rows,
    build_summary_row,
    run_timing_diagnostic,
    sum_duration_per_speaker,
)


class MeetEvalCpwerAlignmentDriftSegmentTimingDiagnosticTest(unittest.TestCase):
    def test_sum_duration_per_speaker(self) -> None:
        totals = sum_duration_per_speaker(
            [
                {"speaker": "SPEAKER_1", "start_time": 0.0, "end_time": 3.0},
                {"speaker": "SPEAKER_1", "start_time": 5.0, "end_time": 7.0},
                {"speaker": "SPEAKER_2", "start_time": 0.0, "end_time": 4.0},
            ]
        )

        self.assertAlmostEqual(totals["SPEAKER_1"], 5.0)
        self.assertAlmostEqual(totals["SPEAKER_2"], 4.0)

    def test_build_speaker_rows_report_delta(self) -> None:
        rows = build_speaker_rows(
            "HeavyOverlap",
            {"SPEAKER_1": 10.0, "SPEAKER_2": 15.0},
            {"SPEAKER_1": 12.0, "SPEAKER_2": 13.0},
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["duration_match"], "False")

    def test_build_summary_row_counts_mismatches(self) -> None:
        summary = build_summary_row(
            "HeavyOverlap",
            [
                {
                    "case_id": "HeavyOverlap",
                    "speaker": "SPEAKER_1",
                    "reference_duration_sec": "10.000",
                    "hypothesis_duration_sec": "12.000",
                    "duration_delta_sec": "2.000",
                    "duration_match": "False",
                }
            ],
        )

        self.assertEqual(summary["mismatched_speaker_count"], "1")

    def test_run_timing_diagnostic_reports_heavy_overlap(self) -> None:
        speaker_rows, summary = run_timing_diagnostic("HeavyOverlap")

        self.assertTrue(speaker_rows)
        self.assertEqual(summary["case_id"], "HeavyOverlap")

    def test_build_receipt_row_marks_complete(self) -> None:
        row = build_receipt_row({"case_id": "HeavyOverlap", "mismatched_speaker_count": "2"})

        self.assertEqual(row["execution_status"], "timing_diagnostic_complete")


if __name__ == "__main__":
    unittest.main()
