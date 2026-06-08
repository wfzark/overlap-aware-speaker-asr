from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_handoff_completion_dashboard_bridge_checklist import (
    build_bridge_checklist_rows,
)


class FrontierOperatorNextActionStatusHandoffCompletionDashboardBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_links_dashboard_to_runbook(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "current_first_frontier": "meeteval_compatibility",
                "dashboard_note": "Status/handoff queue snapshot.",
            }
        )

        self.assertIn("status_handoff_runbook_card", rows[0]["receipt_target"])
        self.assertEqual(rows[0]["bridge_note"], "Status/handoff queue snapshot.")

    def test_build_bridge_checklist_rows_returns_empty_without_dashboard(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
