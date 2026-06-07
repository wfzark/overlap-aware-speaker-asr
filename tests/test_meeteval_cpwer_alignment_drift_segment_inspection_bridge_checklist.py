from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_inspection_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentDriftSegmentInspectionBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_inspection(self) -> None:
        rows = build_bridge_checklist_rows(
            {"case_id": "HeavyOverlap", "inspection_pass": True}
        )

        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")
        self.assertEqual(rows[0]["inspection_status"], "segment_inspection_complete")

    def test_build_bridge_checklist_rows_mark_complete_for_string_true(self) -> None:
        rows = build_bridge_checklist_rows(
            {"case_id": "HeavyOverlap", "inspection_pass": "True"}
        )

        self.assertEqual(rows[0]["inspection_status"], "segment_inspection_complete")

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "case_id": "HeavyOverlap",
                    "inspection_status": "segment_inspection_complete",
                    "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment_drift_segment_inspection.md",
                    "receipt_target": "results/figures/meeteval_cpwer_alignment_drift_segment_handoff_bridge_checklist.md",
                    "checklist_goal": "Verify the segment inspection bridge.",
                    "bridge_note": "Segment inspection status=segment_inspection_complete.",
                    "next_gate": "Confirm this bridge before opening the cpWER segment handoff bridge checklist target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# MeetEval cpWER Alignment Drift Segment Inspection Bridge Checklist", rendered)


if __name__ == "__main__":
    unittest.main()
