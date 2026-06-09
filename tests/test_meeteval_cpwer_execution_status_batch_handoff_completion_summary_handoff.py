from __future__ import annotations

import unittest

from src.meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff import (
    build_handoff_rows,
)


class MeetEvalCpwerExecutionStatusBatchHandoffCompletionSummaryHandoffTest(unittest.TestCase):
    def test_build_handoff_rows_when_queue_complete(self) -> None:
        summary = {
            "queue_status": "queue_complete",
            "complete_handoff_count": "5",
            "total_handoff_count": "5",
        }

        rows = build_handoff_rows(summary)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["handoff_status"], "batch_handoff_completion_handoff_ready")

    def test_build_handoff_rows_empty_when_summary_missing(self) -> None:
        self.assertEqual(build_handoff_rows({}), [])


if __name__ == "__main__":
    unittest.main()
