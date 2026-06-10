from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from src.audio_manifest import read_wav_info, write_manifest


class AudioManifestHelpersTest(unittest.TestCase):
    def test_read_wav_info_reads_duration_and_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "demo.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)

            info = read_wav_info(wav_path)
            self.assertEqual(info["sample_rate"], 16000)
            self.assertEqual(info["channels"], 1)
            self.assertEqual(info["duration_sec"], 1.0)

    def test_write_manifest_writes_csv_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "manifest.csv"
            write_manifest(
                [
                    {
                        "case_id": "Demo",
                        "audio_type": "mixed",
                        "path": "demo.wav",
                        "duration_sec": 1.0,
                        "sample_rate": 16000,
                        "channels": 1,
                        "overlap_level": 0,
                    }
                ],
                output_path,
            )
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("case_id", content)
            self.assertIn("Demo", content)


if __name__ == "__main__":
    unittest.main()
