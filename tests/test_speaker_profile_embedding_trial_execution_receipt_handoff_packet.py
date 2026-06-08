from __future__ import annotations

import unittest

from src.speaker_profile_embedding_trial_execution_receipt_handoff_packet import build_packet_rows


class SpeakerProfileEmbeddingTrialExecutionReceiptHandoffPacketTest(unittest.TestCase):
    def test_build_packet_rows_include_twenty_sections(self) -> None:
        rows = build_packet_rows(
            {
                "case_id": "NoOverlap",
                "readiness_status": "receipt_ready_to_fill",
                "receipt_template_status": "template_only",
            }
        )

        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["section_name"], "receipt_readiness")
        self.assertEqual(rows[1]["section_name"], "receipt_readiness_bridge_checklist")
        self.assertEqual(rows[3]["section_name"], "receipt_open_card_bridge_checklist")
        self.assertEqual(rows[4]["section_name"], "receipt_handoff_packet")
        self.assertEqual(rows[5]["section_name"], "receipt_handoff_packet_bridge_checklist")
        self.assertEqual(rows[6]["section_name"], "receipt_operator_brief")
        self.assertEqual(rows[7]["section_name"], "receipt_operator_brief_bridge")
        self.assertEqual(rows[8]["section_name"], "receipt_operator_brief_bridge_checklist")
        self.assertEqual(rows[9]["section_name"], "receipt_runbook_card")
        self.assertEqual(rows[10]["section_name"], "receipt_runbook_bridge_checklist")
        self.assertEqual(rows[11]["section_name"], "receipt_phase_checkpoint_card")
        self.assertEqual(rows[12]["section_name"], "receipt_phase_checkpoint_bridge_checklist")
        self.assertEqual(rows[13]["section_name"], "receipt_milestone_card")
        self.assertEqual(rows[14]["section_name"], "receipt_milestone_bridge_checklist")
        self.assertEqual(rows[15]["section_name"], "receipt_completion_dashboard")
        self.assertEqual(rows[16]["section_name"], "receipt_completion_dashboard_bridge_checklist")
        self.assertEqual(rows[17]["section_name"], "receipt_status_preflight_bridge_checklist")
        self.assertEqual(rows[18]["section_name"], "receipt_status_reentry_card")
        self.assertEqual(rows[19]["section_name"], "receipt_status_reentry_bridge_checklist")


if __name__ == "__main__":
    unittest.main()
