from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.evaluate_synthetic_benchmark import _display_path, evaluate_sample


class EvaluateSyntheticBenchmarkEvaluateSampleTest(unittest.TestCase):
    def test_display_path_returns_absolute_string_outside_project_root(self) -> None:
        outside = Path("/tmp/outside_repo/sample.json")
        self.assertEqual(_display_path(outside), str(outside))

    def test_evaluate_sample_emits_three_method_cer_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dirs = {
                "raw_dir": base / "raw",
                "speaker_dir": base / "speaker",
                "cleaned_dir": base / "cleaned",
            }
            for directory in dirs.values():
                directory.mkdir(parents=True, exist_ok=True)

            row = {
                "sample_id": "sample_001",
                "tier": "SyntheticNoOverlap",
                "reference_path": "resources/synthetic_overlap/references/SyntheticNoOverlap_01_silver_reference.json",
                "mixed_path": "resources/demo/mixed.wav",
                "spk1_path": "resources/demo/spk1.wav",
                "spk2_path": "resources/demo/spk2.wav",
                "overlap_level_numeric": 0,
            }

            def fake_read_or_transcribe(_model, _audio, _language, output_path, payload_builder, _overwrite):
                if output_path.name.endswith("mixed_whisper.json"):
                    return payload_builder({"text": "你好世", "segments": [], "runtime_sec": 1.0})
                if output_path.name.endswith("spk1_whisper.json"):
                    return payload_builder({"text": "你好", "segments": [], "runtime_sec": 1.0})
                return payload_builder({"text": "世界", "segments": [], "runtime_sec": 1.0})

            with unittest.mock.patch(
                "src.evaluate_synthetic_benchmark.read_or_transcribe",
                side_effect=fake_read_or_transcribe,
            ):
                rows = evaluate_sample(row, object(), "small", "zh", overwrite=True, dirs=dirs)

        methods = {item["method"] for item in rows}
        self.assertEqual(
            methods,
            {"mixed_whisper", "separated_whisper", "separated_whisper_cleaned"},
        )
        self.assertTrue(all(item["sample_id"] == "sample_001" for item in rows))
        self.assertTrue(all("cer" in item for item in rows))


if __name__ == "__main__":
    unittest.main()
