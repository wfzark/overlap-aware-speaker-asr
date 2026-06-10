from __future__ import annotations

import unittest
from typing import Any

from src.router_ablation import STRATEGIES, build_decisions, summarise_gold


def _sample_entry(case_id: str = "NoOverlap") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "overlap_level": 0,
        "mixed_text": "你好世界",
        "separated_text": "你好",
        "cleaned_text": "你好",
        "mixed_segments": [{"text": "你好世界"}],
        "separated_segments": [{"text": "你好"}],
        "cleaned_segments": [{"text": "你好"}],
        "duplicate_removed_count": 0,
        "mixed_runtime_sec": 1.0,
        "separated_runtime_sec": 2.0,
        "cleaned_runtime_sec": 1.5,
        "cleaned_exists": True,
        "cleaned_closer_to_mixed": True,
    }


class RouterAblationSummariseTest(unittest.TestCase):
    def test_build_decisions_fans_out_each_strategy_per_entry(self) -> None:
        entries = [_sample_entry("NoOverlap"), _sample_entry("LightOverlap")]
        decisions = build_decisions(entries, is_gold=True)
        self.assertEqual(len(decisions), len(entries) * len(STRATEGIES))
        self.assertEqual({row["case_id"] for row in decisions}, {"NoOverlap", "LightOverlap"})
        self.assertEqual({row["strategy"] for row in decisions}, set(STRATEGIES))

    def test_summarise_gold_computes_average_cer_and_oracle_gap(self) -> None:
        entries = [_sample_entry("NoOverlap")]
        decisions = build_decisions(entries, is_gold=True)
        cer_lookup = {
            ("NoOverlap", "mixed_whisper"): 0.20,
            ("NoOverlap", "separated_whisper"): 0.10,
            ("NoOverlap", "separated_whisper_cleaned"): 0.12,
        }
        summary = summarise_gold(entries, decisions, cer_lookup)
        fixed_mixed = next(row for row in summary if row["strategy"] == "fixed_mixed_whisper")
        oracle = next(row for row in summary if row["strategy"] == "oracle_best")

        self.assertEqual(fixed_mixed["average_cer"], 0.20)
        self.assertEqual(oracle["average_cer"], 0.10)
        self.assertGreaterEqual(fixed_mixed["gap_to_oracle"], 0.0)


if __name__ == "__main__":
    unittest.main()
