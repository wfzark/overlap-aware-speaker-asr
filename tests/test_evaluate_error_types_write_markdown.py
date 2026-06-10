from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluate_error_types import write_markdown


def _error_row(case_id: str, method: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "method": method,
        "dominant_error_type": "insertion",
        "repetition_count": 2,
        "removed_count_if_cleaned": 1,
        "insertion_count": 5,
        "observation": f"{case_id} observation",
    }


class EvaluateErrorTypesWriteMarkdownTest(unittest.TestCase):
    def test_write_markdown_includes_separated_whisper_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "error_summary.md"
            write_markdown(
                [
                    _error_row("LightOverlap", "separated_whisper"),
                    _error_row("MidOverlap", "separated_whisper"),
                    _error_row("NoOverlap", "mixed_whisper"),
                ],
                md_path,
            )
            content = md_path.read_text(encoding="utf-8")

        self.assertIn("# Error Type Summary", content)
        self.assertIn("LightOverlap", content)
        self.assertIn("separated_whisper", content)
        self.assertIn("insertion-heavy", content)


if __name__ == "__main__":
    unittest.main()
