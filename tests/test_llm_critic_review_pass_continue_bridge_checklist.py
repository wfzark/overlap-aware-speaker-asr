from __future__ import annotations

import unittest

from src.llm_critic_review_pass_continue_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class LlmCriticReviewPassContinueBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_continue_row(self) -> None:
        rows = build_bridge_checklist_rows(
            {"case_id": "NoOverlap", "completed_pass_count": "3"}
        )

        self.assertEqual(rows[0]["case_id"], "NoOverlap")
        self.assertIn("completed_pass_count=3", rows[0]["bridge_note"])

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "case_id": "NoOverlap",
                    "prerequisite_artifact": "results/figures/llm_critic_review_pass_continue.md",
                    "receipt_target": "results/figures/llm_critic_review_pass_continue_receipt.md",
                    "checklist_goal": "Verify the fourth qualitative pass bridge.",
                    "bridge_note": "Continue pass reports completed_pass_count=3.",
                    "next_gate": "Confirm this bridge before opening the critic review pass continue receipt target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# LLM Critic Review Pass Continue Bridge Checklist", rendered)


if __name__ == "__main__":
    unittest.main()
