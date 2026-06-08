from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_handoff_operator_brief_bridge import build_bridge_row


class FrontierOperatorNextActionStatusHandoffOperatorBriefBridgeTest(unittest.TestCase):
    def test_build_bridge_row_links_brief_to_runbook(self) -> None:
        row = build_bridge_row(
            {
                "ready_frontier": "meeteval_compatibility",
                "operator_urgency": "queue_status=queue_complete; ready_lane_count=1; blocked_lane_count=1",
            }
        )

        self.assertEqual(row["reentry_frontier"], "meeteval_compatibility")
        self.assertIn("operator_urgency=queue_status=queue_complete", row["bridge_note"])
        self.assertIn("status_handoff_runbook_card.md", row["receipt_target"])

    def test_build_bridge_row_returns_empty_without_brief(self) -> None:
        row = build_bridge_row({})

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
