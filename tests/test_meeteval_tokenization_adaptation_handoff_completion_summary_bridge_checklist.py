from __future__ import annotations

import unittest

from src.meeteval_tokenization_adaptation_handoff_completion_summary_bridge_checklist import (
    build_bridge_checklist_rows,
)


class MeetEvalTokenizationHandoffCompletionSummaryBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_runbook(self) -> None:
        rows = build_bridge_checklist_rows(
            {"queue_status": "queue_complete", "aligned_count": "5", "total_count": "5"}
        )

        self.assertEqual(rows[0]["queue_status"], "queue_complete")
        self.assertIn("runbook_card", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
