from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_alignment_summary(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "scope": "all_gold_cases",
                "matched_count": 4,
                "case_count": 5,
            }
        )

        self.assertEqual(rows[0]["scope"], "all_gold_cases")
        self.assertIn("matched=4/5", rows[0]["bridge_note"])

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "scope": "all_gold_cases",
                    "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment.md",
                    "receipt_target": "results/figures/meeteval_cpwer_bridge_handoff.md",
                    "checklist_goal": "Verify the cpWER alignment bridge.",
                    "bridge_note": "Alignment reports matched=4/5.",
                    "next_gate": "Confirm this bridge before opening the cpWER bridge handoff target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# MeetEval cpWER Alignment Bridge Checklist", rendered)
        self.assertIn("all_gold_cases", rendered)


if __name__ == "__main__":
    unittest.main()
