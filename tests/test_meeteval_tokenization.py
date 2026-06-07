from __future__ import annotations

import unittest

from src.meeteval_tokenization import (
    count_meeteval_tokens,
    normalize_for_character_wer,
    tokenize_chinese_for_meeteval,
)
from src.meeteval_cpwer_official_execution_tokenization_diagnostic import build_diagnostic_row


class MeetEvalTokenizationTest(unittest.TestCase):
    def test_tokenize_chinese_for_meeteval(self) -> None:
        self.assertEqual(tokenize_chinese_for_meeteval("你好世界"), "你 好 世 界")

    def test_normalize_strips_speaker_tags(self) -> None:
        self.assertEqual(normalize_for_character_wer("[SPEAKER_1]你好"), "你好")

    def test_count_meeteval_tokens(self) -> None:
        self.assertEqual(count_meeteval_tokens("你好"), 2)


class MeetEvalTokenizationDiagnosticTest(unittest.TestCase):
    def test_build_diagnostic_row_identifies_root_cause(self) -> None:
        row = build_diagnostic_row(
            "NoOverlap",
            {"case_id": "NoOverlap", "official_cpwer": "4.0"},
        )
        self.assertEqual(row["root_cause"], "no_whitespace_word_tokenization")
        self.assertEqual(row["diagnostic_status"], "root_cause_identified")


if __name__ == "__main__":
    unittest.main()
