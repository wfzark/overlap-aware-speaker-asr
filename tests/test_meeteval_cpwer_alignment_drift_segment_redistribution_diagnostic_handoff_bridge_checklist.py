from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_redistribution_diagnostic_handoff_bridge_checklist import (
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentDriftSegmentRedistributionDiagnosticHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_cpwer_bridge(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "HeavyOverlap",
                "handoff_status": "redistribution_handoff_ready",
                "redistribution_mismatch_count": "2",
            }
        )

        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")
        self.assertIn("cpWER bridge handoff", rows[0]["next_gate"])


if __name__ == "__main__":
    unittest.main()
