from __future__ import annotations

import unittest

from src.meeteval_cpwer_alignment_drift_segment_granularity_diagnostic_handoff_bridge_checklist import (
    build_bridge_checklist_rows,
)


class MeetEvalCpwerAlignmentDriftSegmentGranularityDiagnosticHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_redistribution_bridge(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "case_id": "HeavyOverlap",
                "handoff_status": "granularity_handoff_ready",
                "mismatched_speaker_count": "1",
            }
        )

        self.assertEqual(rows[0]["case_id"], "HeavyOverlap")
        self.assertIn("redistribution_diagnostic_bridge", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
