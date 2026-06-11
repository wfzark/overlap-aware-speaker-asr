from __future__ import annotations

import unittest

from src.evaluate_cpcer_lite import load_speaker_payload, macro_cer_for_mapping
from src.config import PROJECT_ROOT


class EvaluateCpcerLiteLoadSpeakerTest(unittest.TestCase):
    def test_load_speaker_payload_reads_separated_transcript(self) -> None:
        path, payload, segments = load_speaker_payload("NoOverlap", "separated_whisper")
        self.assertTrue(path.exists())
        self.assertEqual(payload.get("case_id"), "NoOverlap")
        self.assertGreater(len(segments), 0)

    def test_load_speaker_payload_reads_cleaned_transcript(self) -> None:
        path, payload, segments = load_speaker_payload("NoOverlap", "separated_whisper_cleaned")
        self.assertTrue(path.exists())
        self.assertIn("cleaned_segments", payload)
        self.assertGreater(len(segments), 0)

    def test_load_speaker_payload_raises_for_missing_separated_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_speaker_payload("__missing_case__", "separated_whisper")

    def test_load_speaker_payload_raises_for_missing_cleaned_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_speaker_payload("__missing_case__", "separated_whisper_cleaned")

    def test_load_speaker_payload_raises_for_unsupported_method(self) -> None:
        with self.assertRaises(ValueError):
            load_speaker_payload("NoOverlap", "mixed_whisper")

    def test_macro_cer_for_mapping_rejects_unknown_mapping(self) -> None:
        with self.assertRaises(ValueError):
            macro_cer_for_mapping("甲", "乙", "甲", "乙", "diagonal")


if __name__ == "__main__":
    unittest.main()
