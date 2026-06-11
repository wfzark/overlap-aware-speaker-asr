from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.evaluate_synthetic_benchmark import read_or_transcribe


class EvaluateSyntheticBenchmarkReadOrTranscribeTest(unittest.TestCase):
    def test_read_or_transcribe_returns_cached_payload_without_transcribing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample_001_mixed_whisper.json"
            cached = {"sample_id": "sample_001", "text": "cached"}
            output_path.write_text(json.dumps(cached), encoding="utf-8")

            with unittest.mock.patch("src.evaluate_synthetic_benchmark.transcribe_audio") as transcribe:
                payload = read_or_transcribe(
                    model=object(),
                    audio_path=Path(tmp_dir) / "audio.wav",
                    language="zh",
                    output_path=output_path,
                    payload_builder=lambda result: result,
                    overwrite=False,
                )

            transcribe.assert_not_called()
            self.assertEqual(payload["text"], "cached")

    def test_read_or_transcribe_transcribes_when_missing_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample_001_mixed_whisper.json"
            audio_path = Path(tmp_dir) / "audio.wav"
            audio_path.write_text("", encoding="utf-8")
            transcribe_result = {"text": "fresh", "segments": [], "runtime_sec": 1.0}

            with unittest.mock.patch(
                "src.evaluate_synthetic_benchmark.transcribe_audio",
                return_value=transcribe_result,
            ) as transcribe:
                payload = read_or_transcribe(
                    model=object(),
                    audio_path=audio_path,
                    language="zh",
                    output_path=output_path,
                    payload_builder=lambda result: {"sample_id": "sample_001", **result},
                    overwrite=True,
                )

            transcribe.assert_called_once()
            self.assertEqual(payload["text"], "fresh")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
