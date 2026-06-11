from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate_synthetic_routing import write_outputs


class EvaluateSyntheticRoutingWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_writes_decisions_performance_and_summary_markdown(self) -> None:
        decisions = [
            {
                "sample_id": "SyntheticNoOverlap_01",
                "tier": "no_overlap",
                "selected_method": "separated_whisper",
                "decision_rule": "feature_router_v2",
                "mixed_segments_count": 1,
                "separated_segments_count": 2,
                "cleaned_segments_count": 2,
                "mixed_text_length": 10,
                "separated_text_length": 12,
                "cleaned_text_length": 11,
            }
        ]
        performance = [
            {
                "scope": "ALL",
                "strategy": "feature_router_v2",
                "average_cer": 0.2,
                "sample_count": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = {
                "decisions_csv": root / "decisions.csv",
                "decisions_json": root / "decisions.json",
                "performance_csv": root / "performance.csv",
                "performance_json": root / "performance.json",
                "summary_md": root / "summary.md",
            }
            write_outputs(paths, decisions, performance)
            with paths["decisions_csv"].open(encoding="utf-8-sig", newline="") as handle:
                loaded_decisions = list(csv.DictReader(handle))
            loaded_performance = json.loads(paths["performance_json"].read_text(encoding="utf-8"))
            summary = paths["summary_md"].read_text(encoding="utf-8")
            self.assertEqual(loaded_decisions[0]["sample_id"], "SyntheticNoOverlap_01")
            self.assertEqual(loaded_performance[0]["strategy"], "feature_router_v2")
            self.assertIn("# Synthetic Routing Stability Summary", summary)
            self.assertIn("silver references", summary)


if __name__ == "__main__":
    unittest.main()
