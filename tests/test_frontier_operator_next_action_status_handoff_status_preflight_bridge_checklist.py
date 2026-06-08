from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_handoff_status_preflight_bridge_checklist import (
    build_bridge_checklist_rows,
)


class FrontierOperatorNextActionStatusHandoffStatusPreflightBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_links_dashboard_bridge_to_status(self) -> None:
        rows = build_bridge_checklist_rows(
            [
                {
                    "current_first_frontier": "meeteval_compatibility",
                    "next_gate": "Confirm this bridge before opening the status/handoff runbook card target.",
                }
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["current_first_frontier"], "meeteval_compatibility")
        self.assertIn("status_handoff_status.md", rows[0]["receipt_target"])
        self.assertIn("status/handoff runbook card target", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_returns_empty_without_dashboard_bridge(self) -> None:
        rows = build_bridge_checklist_rows([])

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
