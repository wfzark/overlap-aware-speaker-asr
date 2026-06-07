from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_reconciliation_scaffold_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentDriftSegmentReconciliationScaffoldBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_scaffold(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "HeavyOverlap",
                "scaffold_status": "scaffold_only",
                "inspection_status": "segment_inspection_complete",
            }
        )

        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "case_id": "HeavyOverlap",
                    "scaffold_status": "scaffold_only",
                    "inspection_status": "segment_inspection_complete",
                    "prerequisite_artifact": "results/figures/meeteval_cpwer_alignment_drift_segment_reconciliation_scaffold.md",
                    "receipt_target": "results/figures/meeteval_cpwer_alignment_drift_segment_inspection_bridge_checklist.md",
                    "checklist_goal": "Verify the reconciliation scaffold bridge.",
                    "bridge_note": "Reconciliation scaffold remains scaffold_only.",
                    "next_gate": "Confirm this bridge before opening the cpWER segment inspection bridge checklist target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn(
            "# MeetEval cpWER Alignment Drift Segment Reconciliation Scaffold Bridge Checklist",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
