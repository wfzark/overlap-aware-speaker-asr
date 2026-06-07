from __future__ import annotations

import unittest

from src.meeteval_cpwer_execution_preflight_bridge_checklist import build_bridge_checklist_rows


class MeetEvalCpwerExecutionPreflightBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_execution_receipt(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "NoOverlap",
                "preflight_pass": True,
                "hypothesis_source": "separated_whisper",
            }
        )

        self.assertEqual(rows[0]["case_id"], "NoOverlap")
        self.assertIn("execution_receipt", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
