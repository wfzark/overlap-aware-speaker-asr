from __future__ import annotations

import unittest

from src.frontier_operator_next_action_status_bridge_checklist import build_bridge_checklist_rows


class FrontierOperatorNextActionStatusBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_uses_combined_operator_status(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "combined_operator_status": "operator_status_mixed_ready",
                "primary_status_target": "meeteval_compatibility",
            }
        )

        self.assertEqual(rows[0]["combined_operator_status"], "operator_status_mixed_ready")
        self.assertIn("combined_operator_status=operator_status_mixed_ready", rows[0]["bridge_note"])
        self.assertIn("meeteval_compatibility", rows[0]["bridge_note"])

    def test_build_bridge_checklist_rows_defaults_status(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows[0]["combined_operator_status"], "operator_status_unset")


if __name__ == "__main__":
    unittest.main()
