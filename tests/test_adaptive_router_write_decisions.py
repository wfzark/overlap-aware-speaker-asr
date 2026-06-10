from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adaptive_router import DECISION_COLUMNS, write_decisions


def _sample_decision() -> dict[str, object]:
    return {
        "case_id": "FixtureCase",
        "overlap_level": 0,
        "selected_method": "separated_whisper",
        "decision_rule": "overlap_level==0",
        "mixed_segments_count": 3,
        "separated_segments_count": 4,
        "mixed_text_length": 100,
        "separated_text_length": 90,
        "text_length_ratio": 0.9,
        "mixed_runtime_sec": 1.0,
        "separated_runtime_sec": 2.0,
        "runtime_ratio": 2.0,
        "duplicate_removed_count": 0,
        "notes": "fixture",
    }


class AdaptiveRouterWriteDecisionsTest(unittest.TestCase):
    def test_write_decisions_emits_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.adaptive_router.PROJECT_ROOT", root):
                csv_path, json_path = write_decisions([_sample_decision()])

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, DECISION_COLUMNS)
                rows = list(reader)
            self.assertEqual(rows[0]["selected_method"], "separated_whisper")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["case_id"], "FixtureCase")


if __name__ == "__main__":
    unittest.main()
