from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cascade_boundary_bridge import BRIDGE_COLUMNS, write_outputs


class CascadeBoundaryBridgeWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_artifacts(self) -> None:
        row = {
            "strategy": "budget_cascade",
            "case_id": "NoOverlap",
            "overlap_ratio_anchor": 0.0,
            "overlap_level": 0,
            "risk_level": "low",
            "selected_method": "separated_whisper",
            "oracle_method": "separated_whisper",
            "mixed_cer": 0.05,
            "separated_cer": 0.02,
            "separated_cleaned_cer": 0.03,
            "selected_cer": 0.02,
            "oracle_cer": 0.02,
            "delta_cer_separated": -0.03,
            "separation_helps": True,
            "prefers_separation_route": True,
            "cascade_matches_oracle": True,
            "cascade_aligns_with_phase": True,
            "cascade_regret_cer": 0.0,
        }
        summary = [
            {
                "strategy": "budget_cascade",
                "metric": "cascade_oracle_match_rate",
                "value": "1.0",
                "label": "experimental/frontier",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.cascade_boundary_bridge.PROJECT_ROOT", root):
                paths = write_outputs([row], summary)
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
            with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, BRIDGE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
