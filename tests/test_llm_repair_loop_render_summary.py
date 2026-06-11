from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.llm_repair_loop import render_summary


class LlmRepairLoopRenderSummaryTest(unittest.TestCase):
    def test_render_summary_writes_per_case_markdown_table(self) -> None:
        summaries = [
            {
                "case_id": "NoOverlap",
                "original_cer": 0.20,
                "final_cer": 0.15,
                "total_improvement": 0.05,
                "total_iterations": 2,
                "converged": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "results" / "figures").mkdir(parents=True)
            with unittest.mock.patch("src.llm_repair_loop.PROJECT_ROOT", root):
                md_path = render_summary(summaries)
            content = md_path.read_text(encoding="utf-8")
        self.assertIn("# LLM-ASR Repair Loop Results", content)
        self.assertIn("NoOverlap", content)
        self.assertIn("0.0500", content)


if __name__ == "__main__":
    unittest.main()
