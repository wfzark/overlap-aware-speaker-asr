from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_handoff_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentDriftHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_handoff(self) -> None:
        rows = build_bridge_checklist_rows(
            {"case_id": "HeavyOverlap", "drift_severity": "moderate"}
        )

        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "case_id": "HeavyOverlap",
                    "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment_drift_handoff.md",
                    "receipt_target": "results/figures/meeteval_cpwer_alignment_drift_bridge_checklist.md",
                    "checklist_goal": "Verify the drift handoff bridge.",
                    "bridge_note": "Drift handoff remains severity=moderate.",
                    "next_gate": "Confirm this bridge before opening the cpWER alignment drift bridge checklist target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# MeetEval cpWER Alignment Drift Handoff Bridge Checklist", rendered)


if __name__ == "__main__":
    unittest.main()
