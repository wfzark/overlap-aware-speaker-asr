from __future__ import annotations

import unittest
from typing import Any

from src.risk_aware_selector import choose_final_method


def _features(base_method: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "base_v2_method": base_method,
        "base_v2_row": {"overlap_level": 0},
        "cleaned_text": "清理后文本",
        "duplicate_removed_count": 3,
        "cleaned_to_separated_ratio": 0.85,
        "text_length_ratio": 1.2,
    }
    base.update(overrides)
    return base


class RiskAwareSelectorChooseFinalMethodTest(unittest.TestCase):
    def test_choose_final_method_repairs_separated_with_cleaned_transcript(self) -> None:
        method, action = choose_final_method(
            _features("separated_whisper"),
            "medium",
            ["repetition_hallucination_risk"],
        )
        self.assertEqual(method, "separated_whisper_cleaned")
        self.assertIn("repair", action)

    def test_choose_final_method_falls_back_to_mixed_when_cleaned_untrustworthy(self) -> None:
        method, action = choose_final_method(
            _features(
                "separated_whisper",
                cleaned_to_separated_ratio=0.6,
                duplicate_removed_count=12,
            ),
            "high",
            ["repetition_hallucination_risk", "cleaned_over_deletion_risk", "speaker_imbalance_risk"],
        )
        self.assertEqual(method, "mixed_whisper")
        self.assertIn("unstable", action)

    def test_choose_final_method_requests_manual_review_for_high_risk(self) -> None:
        method, action = choose_final_method(
            _features("unknown_method", base_v2_row={"overlap_level": 1}),
            "high",
            ["repetition_hallucination_risk", "length_inflation_risk"],
        )
        self.assertEqual(method, "manual_review")
        self.assertIn("not reliable", action)


if __name__ == "__main__":
    unittest.main()
