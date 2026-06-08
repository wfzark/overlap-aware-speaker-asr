from __future__ import annotations

import unittest

from src.meeteval_cpwer_execution_status_batch_completion_summary import build_completion_summary_row


class MeetEvalCpwerExecutionStatusBatchCompletionSummaryTest(unittest.TestCase):
    def test_build_completion_summary_marks_queue_complete_when_all_complete(self) -> None:
        rows = [
            {"execution_chain_status": "execution_chain_complete"},
            {"execution_chain_status": "execution_chain_complete"},
        ]
        summary = build_completion_summary_row(rows)

        self.assertEqual(summary["queue_status"], "queue_complete")
        self.assertEqual(summary["ready_chain_count"], "2")
        self.assertEqual(summary["pending_chain_count"], "0")

    def test_build_completion_summary_marks_in_progress_when_pending(self) -> None:
        rows = [
            {"execution_chain_status": "execution_chain_ready"},
            {"execution_chain_status": "execution_chain_in_progress"},
        ]
        summary = build_completion_summary_row(rows)

        self.assertEqual(summary["queue_status"], "queue_in_progress")
        self.assertEqual(summary["pending_chain_count"], "1")

    def test_build_completion_summary_marks_ready_to_execute_when_all_ready_but_not_complete(self) -> None:
        rows = [
            {"execution_chain_status": "execution_chain_ready"},
            {"execution_chain_status": "execution_chain_ready"},
        ]
        summary = build_completion_summary_row(rows)

        self.assertEqual(summary["queue_status"], "queue_ready_to_execute")


if __name__ == "__main__":
    unittest.main()
