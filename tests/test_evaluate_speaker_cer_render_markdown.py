from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluate_speaker_cer import render_markdown


def _speaker_row(case_id: str, method: str, macro_cer: float, gap: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "method": method,
        "speaker_macro_cer": macro_cer,
        "speaker_gap": gap,
        "speaker_1_cer": macro_cer,
        "speaker_2_cer": macro_cer,
    }


class EvaluateSpeakerCerRenderMarkdownTest(unittest.TestCase):
    def test_render_markdown_summarizes_average_macro_cer(self) -> None:
        rows = [
            _speaker_row("NoOverlap", "separated_whisper", 0.1, 0.02),
            _speaker_row("NoOverlap", "separated_whisper_cleaned", 0.08, 0.01),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            figure_path = root / "speaker_cer.png"
            csv_path = root / "speaker_cer.csv"
            with patch("src.evaluate_speaker_cer.PROJECT_ROOT", root):
                md_path = render_markdown(rows, figure_path, csv_path)
            content = md_path.read_text(encoding="utf-8")

        self.assertIn("# Speaker-aware CER Summary", content)
        self.assertIn("separated_whisper", content)
        self.assertIn("separated_whisper_cleaned", content)
        self.assertIn("Largest speaker gap", content)


if __name__ == "__main__":
    unittest.main()
