from __future__ import annotations

import unittest

from src.meeteval_tokenization_gain_frontier_fill_runbook_card import build_runbook_card_row


class MeetEvalTokenizationGainFrontierFillRunbookCardTest(unittest.TestCase):
    def test_build_runbook_card_row_promotes_ready_handoff_to_execution_card(self) -> None:
        row = build_runbook_card_row(
            {
                "handoff_status": "tokenization_gain_frontier_fill_handoff_ready",
                "queue_status": "queue_complete",
                "adapted_and_aligned_count": "5",
                "case_count": "5",
                "handoff_goal": "Advance frontier fill execution after tokenization gain handoff completion.",
                "handoff_note": "experimental/frontier tokenization-to-fill handoff only.",
            },
            {
                "recommended_frontier": "meeteval_compatibility",
                "recommended_action": "Execute the real frontier run.",
                "required_evidence": "results/figures/frontier_execution_receipt_fill_execution_handoff.md",
                "completion_signal": "execution_status is no longer template_only",
            },
        )

        self.assertEqual(row["runbook_status"], "tokenization_gain_frontier_fill_runbook_ready")
        self.assertEqual(row["recommended_frontier"], "meeteval_compatibility")
        self.assertEqual(row["adapted_case_ratio"], "5/5")
        self.assertIn("Execute the real frontier run", row["next_action"])
        self.assertIn("not claimed", row["guardrail_note"])

    def test_build_runbook_card_row_pending_when_handoff_is_not_ready(self) -> None:
        row = build_runbook_card_row(
            {"handoff_status": "tokenization_gain_frontier_fill_handoff_pending"},
            {"recommended_frontier": "meeteval_compatibility"},
        )

        self.assertEqual(row["runbook_status"], "tokenization_gain_frontier_fill_runbook_pending")

    def test_build_runbook_card_row_empty_when_inputs_missing(self) -> None:
        self.assertEqual(build_runbook_card_row({}, {"recommended_frontier": "meeteval_compatibility"}), {})
        self.assertEqual(
            build_runbook_card_row({"handoff_status": "tokenization_gain_frontier_fill_handoff_ready"}, {}),
            {},
        )


if __name__ == "__main__":
    unittest.main()
