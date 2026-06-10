from __future__ import annotations

import unittest
from typing import Any

from src.risk_aware_selector import choose_final_method


def _features(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "base_v2_method": "separated_whisper",
        "base_v2_row": {"overlap_level": 3},
        "cleaned_text": "cleaned transcript",
        "duplicate_removed_count": 2,
        "cleaned_to_separated_ratio": 0.9,
        "text_length_ratio": 1.1,
    }
    base.update(overrides)
    return base


class RiskAwareSelectorSeparatedBaseTest(unittest.TestCase):
    def test_choose_final_method_keeps_stable_separated_output(self) -> None:
        method, reason = choose_final_method(_features(), "low", ["low_risk"])
        self.assertEqual(method, "separated_whisper")
        self.assertIn("stable", reason)

    def test_choose_final_method_repairs_with_cleaned_transcript(self) -> None:
        method, reason = choose_final_method(
            _features(),
            "medium",
            ["repetition_hallucination_risk"],
        )
        self.assertEqual(method, "separated_whisper_cleaned")
        self.assertIn("cleaned", reason)


if __name__ == "__main__":
    unittest.main()
