from __future__ import annotations

import unittest
from pathlib import Path

from src.audit_synthetic_benchmark import get_hypothesis_text, get_reference_text, issue_for_row, safe_preview


class AuditSyntheticBenchmarkHelpersTest(unittest.TestCase):
    def test_safe_preview_collapses_whitespace(self) -> None:
        self.assertEqual(safe_preview("你好   世界", limit=10), "你好 世界"[:10])

    def test_get_reference_text_prefers_full_text(self) -> None:
        self.assertEqual(get_reference_text({"full_text": "完整", "text": "短"}), "完整")

    def test_get_hypothesis_text_falls_back_to_segments(self) -> None:
        text = get_hypothesis_text({"segments": [{"text": "片段一"}, {"text": "片段二"}]})
        self.assertEqual(text, "片段一\n片段二")

    def test_issue_for_row_flags_missing_files(self) -> None:
        issue = issue_for_row(
            {"reference_length": 10, "hypothesis_length": 10, "cer": 0.1},
            reference_text="参考",
            hypothesis_text="假设",
            reference_path=Path("/missing/reference.json"),
            hypothesis_path=Path("/missing/hypothesis.json"),
        )
        self.assertEqual(issue, "missing_file")

    def test_issue_for_row_flags_empty_reference_text(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp_dir:
            ref_path = Path(tmp_dir) / "ref.json"
            hyp_path = Path(tmp_dir) / "hyp.json"
            ref_path.write_text("{}", encoding="utf-8")
            hyp_path.write_text("{}", encoding="utf-8")
            issue = issue_for_row(
                {"reference_length": 0, "hypothesis_length": 0, "cer": 0.0},
                reference_text="",
                hypothesis_text="",
                reference_path=ref_path,
                hypothesis_path=hyp_path,
            )
            self.assertEqual(issue, "empty_reference")


if __name__ == "__main__":
    unittest.main()
