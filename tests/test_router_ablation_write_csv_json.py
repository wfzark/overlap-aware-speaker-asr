from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.router_ablation import GOLD_DECISION_COLUMNS, write_csv_json


class RouterAblationWriteCsvJsonTest(unittest.TestCase):
    def test_write_csv_json_writes_matching_csv_and_json_payloads(self) -> None:
        rows = [
            {
                "case_id": "NoOverlap",
                "overlap_level": 0,
                "strategy": "feature_router_v2",
                "selected_method": "separated_whisper",
                "decision_rule": "feature_router_v2",
                "mixed_segments_count": 1,
                "separated_segments_count": 2,
                "cleaned_segments_count": 2,
                "mixed_text_length": 10,
                "separated_text_length": 12,
                "cleaned_text_length": 11,
                "text_length_ratio": 1.2,
                "repetition_count": 0,
                "duplicate_removed_count": 0,
                "mixed_runtime_sec": 1.0,
                "separated_runtime_sec": 2.0,
                "cleaned_runtime_sec": 2.1,
                "runtime_ratio": 2.0,
                "cleaned_closer_to_mixed": "no",
                "notes": "synthetic/silver ablation row",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "gold_decisions.csv"
            json_path = Path(tmp_dir) / "gold_decisions.json"
            write_csv_json(rows, csv_path, json_path, GOLD_DECISION_COLUMNS)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                loaded_csv = list(csv.DictReader(handle))
            loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded_csv[0]["case_id"], "NoOverlap")
        self.assertEqual(loaded_json[0]["selected_method"], "separated_whisper")


if __name__ == "__main__":
    unittest.main()
