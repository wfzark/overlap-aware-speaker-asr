from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from src.build_synthetic_references import build_silver_reference, load_snippet_text
from src.config import PROJECT_ROOT


class BuildSyntheticReferencesSilverTest(unittest.TestCase):
    def test_load_snippet_text_reads_existing_snippet_transcript(self) -> None:
        text, payload = load_snippet_text("con_001.wav")
        self.assertTrue(text)
        self.assertIn("segments", payload)

    def test_load_snippet_text_raises_for_missing_snippet(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_snippet_text("__missing_snippet__.wav")

    def test_build_silver_reference_writes_status_and_turns(self) -> None:
        row = {
            "sample_id": "FixtureSilver_01",
            "tier": "SyntheticNoOverlap",
            "split": "train",
            "overlap_ratio": 0.0,
            "con_source": "con_001.wav",
            "pro_source": "pro_001.wav",
        }
        reference_dir = PROJECT_ROOT / "tmp_test_silver_refs"
        reference_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload, written_path = build_silver_reference(row, reference_dir)
            silver_file = reference_dir / "FixtureSilver_01_silver_reference.json"

            self.assertEqual(payload["status"], "silver_reference")
            self.assertEqual(len(payload["turns"]), 2)
            self.assertIn("[SPEAKER_1]", payload["full_text"])
            self.assertTrue(silver_file.exists())
            self.assertIn("FixtureSilver_01_silver_reference.json", written_path)
            on_disk = json.loads(silver_file.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["sample_id"], "FixtureSilver_01")
        finally:
            shutil.rmtree(reference_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
