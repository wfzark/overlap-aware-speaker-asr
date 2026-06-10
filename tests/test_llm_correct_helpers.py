from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.llm_correct import read_csv_rows


class LlmCorrectHelpersTest(unittest.TestCase):
    def test_read_csv_rows_returns_empty_for_missing_file(self) -> None:
        missing = Path(tempfile.gettempdir()) / "missing_llm_correct_rows.csv"
        self.assertEqual(read_csv_rows(missing), [])

    def test_read_csv_rows_reads_dict_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "rows.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["case_id", "risk_level"])
                writer.writeheader()
                writer.writerow({"case_id": "NoOverlap", "risk_level": "low"})
            rows = read_csv_rows(csv_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "NoOverlap")


if __name__ == "__main__":
    unittest.main()
