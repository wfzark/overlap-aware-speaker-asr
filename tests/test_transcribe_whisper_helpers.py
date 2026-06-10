from __future__ import annotations

import unittest
from pathlib import Path

from src.config import load_config
from src.transcribe_whisper import (
    find_case,
    get_model_name,
    get_transcript_text_length,
    preview,
    resolve_audio_path,
    select_cases,
    transcript_path_for,
)


class TranscribeWhisperHelpersTest(unittest.TestCase):
    def test_preview_collapses_whitespace_and_truncates(self) -> None:
        long_text = "你好 " * 50
        self.assertLessEqual(len(preview(long_text, limit=20)), 20)
        self.assertEqual(preview("a\n\tb  c"), "a b c")

    def test_transcript_path_for_mixed_and_separated_modes(self) -> None:
        mixed = transcript_path_for("NoOverlap", "mixed")
        separated = transcript_path_for("NoOverlap", "separated_mixed")
        self.assertTrue(mixed.name.endswith("_mixed_whisper.json"))
        self.assertTrue(separated.name.endswith("_mixed_whisper.json"))
        self.assertIn("transcripts_raw", mixed.as_posix())

    def test_get_model_name_uses_config_default(self) -> None:
        config = load_config()
        self.assertEqual(get_model_name(config), config.get("asr", {}).get("whisper_model", "small"))

    def test_get_model_name_rejects_non_lightweight_models(self) -> None:
        config = load_config()
        with self.assertRaises(ValueError):
            get_model_name(config, override="large")

    def test_find_case_returns_matching_audio_case(self) -> None:
        config = load_config()
        case = find_case(config, "NoOverlap")
        self.assertEqual(case["id"], "NoOverlap")

    def test_get_transcript_text_length_counts_characters(self) -> None:
        self.assertEqual(get_transcript_text_length({"text": "你好世界"}), 4)
        self.assertEqual(get_transcript_text_length({}, "full_text"), 0)

    def test_select_cases_returns_single_case_or_all(self) -> None:
        config = load_config()
        self.assertEqual(len(select_cases(config, "NoOverlap")), 1)
        self.assertGreater(len(select_cases(config, "all")), 1)

    def test_resolve_audio_path_points_to_mixed_audio(self) -> None:
        config = load_config()
        case = find_case(config, "NoOverlap")
        audio_path = resolve_audio_path(config, case, "mixed")
        self.assertTrue(audio_path.name.endswith(".wav"))


if __name__ == "__main__":
    unittest.main()
