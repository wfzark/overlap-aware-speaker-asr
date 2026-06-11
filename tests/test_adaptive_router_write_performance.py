from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adaptive_router import PERFORMANCE_COLUMNS, write_performance


class AdaptiveRouterWritePerformanceTest(unittest.TestCase):
    def test_write_performance_emits_csv_and_json(self) -> None:
        rows = [
            {"strategy": "fixed_mixed_whisper", "average_cer": 0.21},
            {"strategy": "rule_router", "average_cer": 0.15},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.adaptive_router.PROJECT_ROOT", root):
                csv_path, json_path = write_performance(rows)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, PERFORMANCE_COLUMNS)
                loaded = list(reader)
            self.assertEqual(loaded[0]["strategy"], "fixed_mixed_whisper")
            self.assertEqual(loaded[1]["average_cer"], "0.15")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[1]["strategy"], "rule_router")


if __name__ == "__main__":
    unittest.main()
