from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluate_synthetic_benchmark import load_manifest_rows, select_rows


class EvaluateSyntheticBenchmarkManifestTest(unittest.TestCase):
    def test_load_manifest_rows_reads_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "tier"])
                writer.writeheader()
                writer.writerow({"sample_id": "sample_001", "tier": "SyntheticNoOverlap"})
            rows = load_manifest_rows(manifest)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "sample_001")

    def test_select_rows_returns_all_when_case_is_all(self) -> None:
        rows = [{"sample_id": "a"}, {"sample_id": "b"}]
        self.assertEqual(select_rows(rows, "all"), rows)


if __name__ == "__main__":
    unittest.main()
