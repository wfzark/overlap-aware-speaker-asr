from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_reconciliation_handoff import (
    build_handoff_receipt_rows,
    build_handoff_row,
)


class MeetEvalCpwerAlignmentDriftSegmentReconciliationHandoffTest(unittest.TestCase):
    def test_build_handoff_row_targets_reconciliation(self) -> None:
        row = build_handoff_row(
            {
                "case_id": "HeavyOverlap",
                "scaffold_status": "scaffold_only",
                "reconciliation_target": "Speaker-attributed segment reconciliation for HeavyOverlap.",
                "expected_evidence": "results/tables/meeteval_hypothesis_segments.jsonl",
            }
        )

        self.assertEqual(row["handoff_status"], "reconciliation_handoff_ready")
        self.assertIn("HeavyOverlap", row["handoff_goal"])

    def test_build_handoff_receipt_rows_mark_documented(self) -> None:
        rows = build_handoff_receipt_rows({"case_id": "HeavyOverlap"})

        self.assertEqual(rows[0]["execution_status"], "handoff_documented")


if __name__ == "__main__":
    unittest.main()
