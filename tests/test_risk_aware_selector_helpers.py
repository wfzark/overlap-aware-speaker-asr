from __future__ import annotations

import unittest

from src.risk_aware_selector import (
    adjacent_repeat_count,
    aggregate_speaker_text,
    repeat_phrase_count,
    speaker_lengths_from_segments,
)


class RiskAwareSelectorHelpersTest(unittest.TestCase):
    def test_aggregate_speaker_text_joins_matching_segments(self) -> None:
        segments = [
            {"speaker": "SPEAKER_1", "text": "你好"},
            {"speaker": "SPEAKER_2", "text": "世界"},
            {"speaker": "SPEAKER_1", "text": "测试"},
        ]
        self.assertEqual(aggregate_speaker_text(segments, "SPEAKER_1"), "你好测试")

    def test_repeat_phrase_count_detects_repeated_clause(self) -> None:
        text = "同意这个观点\n同意这个观点\n其他"
        self.assertGreaterEqual(repeat_phrase_count(text), 1)

    def test_adjacent_repeat_count_detects_back_to_back_duplicates(self) -> None:
        segments = [
            {"text": "重复句子"},
            {"text": "重复句子"},
            {"text": "不同句子"},
        ]
        self.assertEqual(adjacent_repeat_count(segments), 1)

    def test_speaker_lengths_from_segments_returns_pair(self) -> None:
        segments = [
            {"speaker": "SPEAKER_1", "text": "abc"},
            {"speaker": "SPEAKER_2", "text": "defghij"},
        ]
        self.assertEqual(speaker_lengths_from_segments(segments), (3, 7))


if __name__ == "__main__":
    unittest.main()
