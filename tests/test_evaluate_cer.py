from __future__ import annotations

import unittest

from src.evaluate_cer import levenshtein_distance, normalize_text


class EvaluateCerNormalizationTest(unittest.TestCase):
    def test_normalize_text_strips_speaker_labels(self) -> None:
        self.assertEqual(normalize_text("[SPEAKER_1] 你好"), "你好")

    def test_normalize_text_removes_punctuation(self) -> None:
        self.assertEqual(normalize_text("你好，世界！"), "你好世界")


class EvaluateCerLevenshteinTest(unittest.TestCase):
    def test_levenshtein_distance_counts_single_deletion(self) -> None:
        self.assertEqual(levenshtein_distance("你好世界", "你好世"), 1)


if __name__ == "__main__":
    unittest.main()
