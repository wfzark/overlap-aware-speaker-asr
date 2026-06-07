from __future__ import annotations

import unittest

from src.meeteval_cpwer_execution_preflight import build_receipt_rows, run_preflight


class MeetEvalCpwerExecutionPreflightTest(unittest.TestCase):
    def test_build_receipt_rows_marks_preflight_complete_on_pass(self) -> None:
        rows = build_receipt_rows({"case_id": "NoOverlap", "preflight_pass": True, "hypothesis_source": "test"})

        self.assertEqual(rows[0]["execution_status"], "preflight_complete")

    def test_build_receipt_rows_marks_preflight_failed_on_fail(self) -> None:
        rows = build_receipt_rows({"case_id": "NoOverlap", "preflight_pass": False, "hypothesis_source": "test"})

        self.assertEqual(rows[0]["execution_status"], "preflight_failed")

    def test_run_preflight_returns_required_fields(self) -> None:
        row = run_preflight("NoOverlap", {"handoff_status": "execution_handoff_ready", "scaffold_status": "scaffold_only"})

        self.assertEqual(row["case_id"], "NoOverlap")
        self.assertIn("preflight_pass", row)
        self.assertIn("preflight_note", row)


if __name__ == "__main__":
    unittest.main()
