from __future__ import annotations

import unittest

from src.frontier_operator_next_action_operator_brief import build_operator_brief_row


class FrontierOperatorNextActionOperatorBriefTest(unittest.TestCase):
    def test_build_operator_brief_row_summarizes_ready_and_blocked_lanes(self) -> None:
        row = build_operator_brief_row(
            {
                "coordination_state": "mixed_ready_state",
                "go_count": "4",
                "no_go_count": "1",
            },
            [
                {
                    "action_lane": "ready_lane",
                    "frontier_name": "meeteval_compatibility",
                    "operator_action": "Fill the official receipt with real evidence.",
                    "target_artifact": "results/tables/meeteval_cpwer_execution_receipt.json",
                },
                {
                    "action_lane": "blocked_lane",
                    "frontier_name": "external_validation",
                    "operator_action": "Record the license confirmation decision.",
                    "target_artifact": "results/tables/external_validation_license_confirmation_receipt_bridge.json",
                },
            ],
        )

        self.assertEqual(row["ready_frontier"], "meeteval_compatibility")
        self.assertIn("Fill the official receipt", row["ready_action"])
        self.assertEqual(row["blocked_frontier"], "external_validation")
        self.assertIn("mixed_ready_state", row["operator_urgency"])

    def test_build_operator_brief_row_returns_empty_when_no_operator_rows(self) -> None:
        row = build_operator_brief_row({}, [])

        self.assertEqual(row, {})


if __name__ == "__main__":
    unittest.main()
