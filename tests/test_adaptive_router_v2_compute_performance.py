from __future__ import annotations

import unittest

from src.adaptive_router_v2 import compute_performance


class AdaptiveRouterV2ComputePerformanceTest(unittest.TestCase):
    def test_compute_performance_averages_fixed_baselines(self) -> None:
        cer_lookup = {
            ("NoOverlap", "mixed_whisper"): 0.2,
            ("NoOverlap", "separated_whisper"): 0.05,
            ("NoOverlap", "separated_whisper_cleaned"): 0.08,
        }
        decisions = [{"case_id": "NoOverlap", "selected_method": "separated_whisper"}]
        rows = compute_performance(cer_lookup, decisions, "case_id", ["NoOverlap"])
        by_strategy = {row["strategy"]: row["average_cer"] for row in rows}
        self.assertAlmostEqual(by_strategy["fixed_mixed_whisper"], 0.2)
        self.assertAlmostEqual(by_strategy["fixed_separated_whisper"], 0.05)
        self.assertAlmostEqual(by_strategy["oracle_best"], 0.05)


if __name__ == "__main__":
    unittest.main()
