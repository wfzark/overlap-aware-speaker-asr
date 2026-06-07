from __future__ import annotations

import unittest

from src.llm_critic_review_pass_final_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class LlmCriticReviewPassFinalBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_final_row(self) -> None:
        rows = build_bridge_checklist_rows(
            {"case_id": "OppositeOverlap", "completed_pass_count": "4"}
        )

        self.assertEqual(rows[0]["case_id"], "OppositeOverlap")
        self.assertIn("completion_summary", rows[0]["receipt_target"])

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "case_id": "OppositeOverlap",
                    "prerequisite_artifact": "results/figures/llm_critic_review_pass_final.md",
                    "receipt_target": "results/figures/llm_critic_review_pass_completion_summary.md",
                    "checklist_goal": "Verify the final qualitative pass bridge.",
                    "bridge_note": "Final pass reports completed_pass_count=4.",
                    "next_gate": "Confirm this bridge before opening the critic review pass completion summary target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# LLM Critic Review Pass Final Bridge Checklist", rendered)


if __name__ == "__main__":
    unittest.main()
