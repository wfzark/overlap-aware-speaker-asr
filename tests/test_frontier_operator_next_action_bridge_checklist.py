from __future__ import annotations

import unittest

from src.frontier_operator_next_action_bridge_checklist import build_bridge_checklist_rows


class FrontierOperatorNextActionBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_preserves_lane_order(self) -> None:
        rows = build_bridge_checklist_rows(
            [
                {
                    "action_lane": "ready_lane",
                    "frontier_name": "meeteval_compatibility",
                    "go_no_go_state": "go",
                    "target_artifact": "results/tables/meeteval_cpwer_execution_receipt.json",
                },
                {
                    "action_lane": "blocked_lane",
                    "frontier_name": "external_validation",
                    "go_no_go_state": "no_go",
                    "target_artifact": "results/tables/external_validation_license_confirmation_receipt_bridge.json",
                },
            ]
        )

        self.assertEqual([row["action_lane"] for row in rows], ["ready_lane", "blocked_lane"])
        self.assertEqual(rows[0]["target_artifact"], "results/tables/meeteval_cpwer_execution_receipt.json")
        self.assertEqual(rows[1]["target_artifact"], "results/tables/external_validation_license_confirmation_receipt_bridge.json")

    def test_build_bridge_checklist_rows_returns_empty_for_no_operator_rows(self) -> None:
        rows = build_bridge_checklist_rows([])

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
