from __future__ import annotations

import unittest

from src.meeteval_cpwer_bridge_handoff_bridge_checklist import build_bridge_checklist_rows


class MeetEvalCpwerBridgeHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_meeteval_execution(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "NoOverlap",
                "bridge_status": "cpwer_bridge_complete",
                "cpwer_bridge_lite": "0.12",
            }
        )

        self.assertEqual(rows[0]["case_id"], "NoOverlap")
        self.assertIn("MeetEval", rows[0]["next_gate"])


if __name__ == "__main__":
    unittest.main()
