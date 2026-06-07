from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_timing_diagnostic_handoff import (
    build_handoff_receipt_rows,
    build_handoff_row,
)


class MeetEvalCpwerAlignmentDriftSegmentTimingDiagnosticHandoffTest(unittest.TestCase):
    def test_build_handoff_row_targets_granularity_diagnostic(self) -> None:
        row = build_handoff_row(
            {
                "case_id": "HeavyOverlap",
                "mismatched_speaker_count": "1",
                "dominant_blocker": "SPEAKER_1 delta=-2.360s",
            }
        )

        self.assertEqual(row["case_id"], "HeavyOverlap")
        self.assertIn("granularity_diagnostic", row["granularity_diagnostic_target"])

    def test_build_handoff_receipt_rows_document_handoff(self) -> None:
        rows = build_handoff_receipt_rows({"case_id": "HeavyOverlap"})

        self.assertEqual(rows[0]["execution_status"], "handoff_documented")


if __name__ == "__main__":
    unittest.main()
