from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.compare_mixed_vs_separated import read_existing_rows


class CompareMixedVsSeparatedReadExistingTest(unittest.TestCase):
    def test_read_existing_rows_returns_empty_for_missing_file(self) -> None:
        self.assertEqual(read_existing_rows(Path("/tmp/__missing_compare_table__.csv")), [])

    def test_read_existing_rows_reads_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "compare.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "model"])
                writer.writeheader()
                writer.writerow({"case_id": "FixtureCase", "model": "tiny"})

            rows = read_existing_rows(csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "FixtureCase")


if __name__ == "__main__":
    unittest.main()
