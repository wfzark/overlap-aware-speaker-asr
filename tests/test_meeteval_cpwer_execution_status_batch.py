from __future__ import annotations

import unittest

from src.meeteval_cpwer_execution_status_batch import build_status_row, build_status_rows


class MeetEvalCpwerExecutionStatusBatchTest(unittest.TestCase):
    def test_build_status_row_marks_chain_ready_for_batch_scaffold(self) -> None:
        row = build_status_row(
            {"case_id": "NoOverlap", "preflight_pass": True},
            {"case_id": "NoOverlap", "scaffold_status": "receipt_batch_scaffold_only"},
            "template_only",
        )

        self.assertEqual(row["execution_chain_status"], "execution_chain_ready")
        self.assertEqual(row["scope"], "meeteval_cpwer_execution_chain_batch")

    def test_build_status_rows_cover_all_gold_cases(self) -> None:
        rows = build_status_rows([], [], {})

        self.assertEqual(len(rows), 5)

    def test_build_status_rows_mark_ready_when_preflight_and_scaffold_align(self) -> None:
        rows = build_status_rows(
            [{"case_id": "NoOverlap", "preflight_pass": True}],
            [{"case_id": "NoOverlap", "scaffold_status": "receipt_batch_scaffold_only"}],
            {"NoOverlap": "template_only"},
        )

        self.assertEqual(rows[0]["execution_chain_status"], "execution_chain_ready")

    def test_build_status_row_marks_complete_when_official_receipt_written(self) -> None:
        row = build_status_row(
            {"case_id": "NoOverlap", "preflight_pass": True},
            {"case_id": "NoOverlap", "scaffold_status": "receipt_batch_scaffold_only"},
            "official_cpwer_narrow_dry_run_complete",
        )

        self.assertEqual(row["execution_receipt_status"], "official_cpwer_narrow_dry_run_complete")
        self.assertEqual(row["execution_chain_status"], "execution_chain_complete")


if __name__ == "__main__":
    unittest.main()
