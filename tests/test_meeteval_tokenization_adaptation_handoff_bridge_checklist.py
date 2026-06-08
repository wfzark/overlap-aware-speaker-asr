from __future__ import annotations

import unittest

from src.meeteval_tokenization_adaptation_handoff_bridge_checklist import build_bridge_checklist_rows


class MeetEvalTokenizationAdaptationHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_evidence_receipt(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "handoff_status": "tokenization_adaptation_handoff_ready",
                "aligned_count": "5",
                "total_count": "5",
            }
        )

        self.assertEqual(rows[0]["handoff_status"], "tokenization_adaptation_handoff_ready")
        self.assertIn("evidence_receipt", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
