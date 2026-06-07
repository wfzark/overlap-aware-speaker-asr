from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_redistribution_diagnostic_bridge_checklist import (
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentDriftSegmentRedistributionDiagnosticBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_granularity_handoff_bridge(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "HeavyOverlap",
                "redistribution_mismatch_count": "2",
                "dominant_blocker": "SPEAKER_2 hypothesis_split",
            }
        )

        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")
        self.assertIn("granularity_diagnostic_handoff_bridge", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
