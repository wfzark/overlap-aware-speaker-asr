from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_reconciliation_scaffold import (
    build_scaffold_receipt_rows,
    build_scaffold_row,
)


class MeetEvalCpwerAlignmentDriftSegmentReconciliationScaffoldTest(unittest.TestCase):
    def test_build_scaffold_row_uses_inspection(self) -> None:
        row = build_scaffold_row(
            {"case_id": "HeavyOverlap", "inspection_pass": True}
        )

        self.assertEqual(row["case_id"], "HeavyOverlap")
        self.assertEqual(row["inspection_status"], "segment_inspection_complete")

    def test_build_scaffold_receipt_rows_mark_documented(self) -> None:
        rows = build_scaffold_receipt_rows({"case_id": "HeavyOverlap"})

        self.assertEqual(rows[0]["execution_status"], "scaffold_documented")


if __name__ == "__main__":
    unittest.main()
