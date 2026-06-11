from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adaptive_router import update_summary_md


def _sample_decision() -> dict[str, object]:
    return {
        "case_id": "FixtureCase",
        "overlap_level": 0,
        "selected_method": "separated_whisper",
        "decision_rule": "overlap_level==0",
    }


class AdaptiveRouterUpdateSummaryMdTest(unittest.TestCase):
    def test_update_summary_md_writes_decision_and_performance_sections(self) -> None:
        decisions = [_sample_decision()]
        performance_rows = [
            {"strategy": "fixed_mixed_whisper", "average_cer": 0.21},
            {"strategy": "rule_router", "average_cer": 0.15},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.adaptive_router.PROJECT_ROOT", root):
                md_path = update_summary_md(decisions, performance_rows)

            self.assertTrue(md_path.exists())
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("Current Results Summary", markdown)
            self.assertIn("FixtureCase", markdown)
            self.assertIn("fixed_mixed_whisper: 0.210000", markdown)
            self.assertIn("rule_router: 0.150000", markdown)


if __name__ == "__main__":
    unittest.main()
