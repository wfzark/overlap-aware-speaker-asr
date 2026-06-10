from __future__ import annotations

import unittest

from src.evaluate_cer import (
    compute_cer,
    levenshtein_distance,
    list_verified_cases,
    normalize_text,
    sanitize_existing_rows,
    upsert_row,
)


class EvaluateCerHelpersTest(unittest.TestCase):
    def test_normalize_text_strips_speaker_tags_and_punctuation(self) -> None:
        self.assertEqual(normalize_text("[SPEAKER_1] 你好，世界"), "你好世界")

    def test_compute_cer_returns_edit_distance_ratio(self) -> None:
        result = compute_cer("你好世界", "你好世")
        self.assertEqual(result["edit_distance"], 1)
        self.assertEqual(result["reference_length"], 4)
        self.assertEqual(result["cer"], 0.25)

    def test_list_verified_cases_returns_five_gold_cases(self) -> None:
        cases = list_verified_cases()
        self.assertEqual(len(cases), 5)
        self.assertIn("NoOverlap", cases)

    def test_levenshtein_distance_counts_minimum_edits(self) -> None:
        self.assertEqual(levenshtein_distance("你好世界", "你好世"), 1)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_sanitize_existing_rows_skips_incomplete_rows(self) -> None:
        rows = sanitize_existing_rows(
            [
                {"case_id": "Demo", "method": "mixed_whisper"},
                {"case_id": "", "method": "mixed_whisper"},
            ]
        )
        self.assertEqual(len(rows), 1)

    def test_upsert_row_replaces_matching_case_method(self) -> None:
        rows = [{"case_id": "A", "method": "mixed_whisper", "cer": 0.1}]
        updated = upsert_row(rows, {"case_id": "A", "method": "mixed_whisper", "cer": 0.2})
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["cer"], 0.2)


if __name__ == "__main__":
    unittest.main()
