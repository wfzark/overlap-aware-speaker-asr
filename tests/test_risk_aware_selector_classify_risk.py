from __future__ import annotations

import unittest
from typing import Any

from src.risk_aware_selector import classify_risk


def _features(**overrides: Any) -> dict[str, Any]:
    base = {
        "repetition_count": 0,
        "text_length_ratio": 1.1,
        "speaker_length_imbalance": 0.1,
        "duplicate_removed_count": 0,
        "cleaned_text": "",
        "cleaned_to_separated_ratio": 1.0,
        "method_disagreement_score": 0.1,
    }
    base.update(overrides)
    return base


class RiskAwareSelectorClassifyRiskTest(unittest.TestCase):
    def test_classify_risk_returns_medium_for_single_reason(self) -> None:
        risk_level, reasons = classify_risk(_features(repetition_count=4))
        self.assertEqual(risk_level, "medium")
        self.assertEqual(reasons, ["repetition_hallucination_risk"])

    def test_classify_risk_returns_high_for_multiple_reasons(self) -> None:
        risk_level, reasons = classify_risk(
            _features(
                repetition_count=7,
                text_length_ratio=3.0,
                method_disagreement_score=0.5,
            )
        )
        self.assertEqual(risk_level, "high")
        self.assertGreaterEqual(len(reasons), 3)

    def test_classify_risk_flags_cleaned_over_deletion_with_speaker_imbalance(self) -> None:
        risk_level, reasons = classify_risk(
            _features(
                duplicate_removed_count=10,
                cleaned_text="短文本",
                cleaned_to_separated_ratio=0.7,
                speaker_length_imbalance=0.5,
            )
        )
        self.assertEqual(risk_level, "high")
        self.assertIn("cleaned_over_deletion_risk", reasons)
        self.assertIn("speaker_imbalance_risk", reasons)


if __name__ == "__main__":
    unittest.main()
