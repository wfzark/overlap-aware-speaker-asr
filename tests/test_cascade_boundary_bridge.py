from __future__ import annotations

import unittest

from src.cascade_boundary_bridge import build_bridge_rows, build_summary_rows


class CascadeBoundaryBridgeTest(unittest.TestCase):
    def test_build_bridge_rows_budget_cascade_respects_overlap(self) -> None:
        cases = [
            {"case_id": "NoOverlap", "overlap_level": 0, "risk_level": "low"},
            {"case_id": "LightOverlap", "overlap_level": 1, "risk_level": "low"},
        ]
        decisions: dict[str, dict[str, str]] = {}
        cer_by_case = {
            "NoOverlap": {
                "mixed_whisper": 0.05,
                "separated_whisper": 0.02,
                "separated_whisper_cleaned": 0.03,
            },
            "LightOverlap": {
                "mixed_whisper": 0.10,
                "separated_whisper": 0.30,
                "separated_whisper_cleaned": 0.25,
            },
        }
        rows = build_bridge_rows(cases, decisions, cer_by_case, strategies=["budget_cascade"])
        by_case = {row["case_id"]: row for row in rows}
        self.assertEqual(by_case["NoOverlap"]["selected_method"], "separated_whisper")
        self.assertEqual(by_case["LightOverlap"]["selected_method"], "mixed_whisper")
        self.assertTrue(by_case["LightOverlap"]["cascade_aligns_with_phase"])

    def test_build_summary_rows_per_strategy(self) -> None:
        rows = [
            {
                "strategy": "router_v2_costed",
                "cascade_matches_oracle": True,
                "cascade_aligns_with_phase": True,
                "cascade_regret_cer": 0.0,
            },
            {
                "strategy": "router_v2_costed",
                "cascade_matches_oracle": False,
                "cascade_aligns_with_phase": True,
                "cascade_regret_cer": 0.1,
            },
        ]
        summary = build_summary_rows(rows)
        metrics = {
            (row["strategy"], row["metric"]): row["value"]
            for row in summary
            if row["strategy"] == "router_v2_costed"
        }
        self.assertEqual(metrics[("router_v2_costed", "cascade_oracle_match_rate")], "0.5")
        self.assertEqual(metrics[("router_v2_costed", "average_cascade_regret_cer")], "0.05")


if __name__ == "__main__":
    unittest.main()
