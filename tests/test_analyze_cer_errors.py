from __future__ import annotations

import unittest

from src.analyze_cer_errors import extract_hypothesis_text, find_repeated_phrases


class AnalyzeCerErrorsPhraseDetectionTest(unittest.TestCase):
    def test_find_repeated_phrases_detects_repeated_clause(self) -> None:
        text = "你好世界\n你好世界\n其他内容"
        phrases = find_repeated_phrases(text)
        types = {item["type"] for item in phrases}
        self.assertIn("repeated_clause", types)

    def test_find_repeated_phrases_detects_high_frequency_chunk(self) -> None:
        text = "abababababab"
        phrases = find_repeated_phrases(text)
        types = {item["type"] for item in phrases}
        self.assertIn("high_frequency_chunk", types)


class AnalyzeCerErrorsHypothesisExtractionTest(unittest.TestCase):
    def test_extract_hypothesis_text_uses_text_field_for_mixed(self) -> None:
        payload = {"text": "mixed transcript", "full_text": "ignored"}
        self.assertEqual(extract_hypothesis_text(payload, "mixed_whisper"), "mixed transcript")

    def test_extract_hypothesis_text_uses_full_text_for_separated(self) -> None:
        payload = {"text": "ignored", "full_text": "separated transcript"}
        self.assertEqual(
            extract_hypothesis_text(payload, "separated_whisper"),
            "separated transcript",
        )


if __name__ == "__main__":
    unittest.main()
