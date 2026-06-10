from __future__ import annotations

import json
import unittest

from src.config import PROJECT_ROOT
from src.postprocess_transcript import clean_case


class PostprocessTranscriptCleanCaseTest(unittest.TestCase):
    def test_clean_case_skips_existing_output_without_overwrite(self) -> None:
        output_path = clean_case("NoOverlap", overwrite=False)
        expected = (
            PROJECT_ROOT
            / "results"
            / "transcripts_postprocessed"
            / "NoOverlap_separated_speaker_transcript_cleaned.json"
        )
        self.assertEqual(output_path, expected)

    def test_clean_case_writes_duplicate_suppression_payload_with_overwrite(self) -> None:
        output_path = clean_case("NoOverlap", overwrite=True)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["case_id"], "NoOverlap")
        self.assertEqual(payload["method"], "duplicate_suppression")
        self.assertIn("cleaned_segments", payload)
        self.assertIn("removed_count", payload)


if __name__ == "__main__":
    unittest.main()
