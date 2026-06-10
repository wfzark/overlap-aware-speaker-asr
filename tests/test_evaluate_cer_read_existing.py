from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import PROJECT_ROOT
from src.evaluate_cer import levenshtein_distance, list_verified_cases, load_reference, read_existing_rows


class EvaluateCerReadExistingTest(unittest.TestCase):
    def test_read_existing_rows_parses_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "rows.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "method", "cer"])
                writer.writeheader()
                writer.writerow({"case_id": "Demo", "method": "mixed_whisper", "cer": "0.1"})
            rows = read_existing_rows(csv_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "Demo")

    def test_read_existing_rows_parses_json_dict_with_rows_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "rows.json"
            json_path.write_text(
                json.dumps({"rows": [{"case_id": "A", "method": "mixed_whisper"}]}),
                encoding="utf-8",
            )
            rows = read_existing_rows(json_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "A")

    def test_read_existing_rows_parses_json_dict_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "row.json"
            json_path.write_text(
                json.dumps({"case_id": "Solo", "method": "mixed_whisper"}),
                encoding="utf-8",
            )
            rows = read_existing_rows(json_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "Solo")

    def test_read_existing_rows_skips_non_dict_json_list_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "rows.json"
            json_path.write_text(json.dumps([{"case_id": "A", "method": "mixed_whisper"}, "skip"]), encoding="utf-8")
            rows = read_existing_rows(json_path)
        self.assertEqual(len(rows), 1)

    def test_read_existing_rows_returns_empty_for_unknown_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = Path(tmp_dir) / "rows.txt"
            txt_path.write_text("ignored", encoding="utf-8")
            self.assertEqual(read_existing_rows(txt_path), [])

    def test_levenshtein_distance_handles_nonempty_left_operand(self) -> None:
        self.assertEqual(levenshtein_distance("abc", ""), 3)

    def test_load_reference_raises_when_reference_file_missing(self) -> None:
        ref_path = PROJECT_ROOT / "references" / "reference_transcripts.json"
        backup = ref_path.read_bytes()
        try:
            ref_path.unlink()
            with self.assertRaises(FileNotFoundError):
                load_reference("NoOverlap")
        finally:
            ref_path.write_bytes(backup)

    def test_load_reference_rejects_missing_case(self) -> None:
        with self.assertRaises(KeyError):
            load_reference("__missing_case__")

    def test_load_reference_rejects_unverified_status(self) -> None:
        ref_path = PROJECT_ROOT / "references" / "reference_transcripts.json"
        original = ref_path.read_text(encoding="utf-8-sig")
        try:
            payload = json.loads(original)
            payload["TempUnverified"] = {
                "status": "draft",
                "full_text": "临时",
            }
            ref_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reference("TempUnverified")
        finally:
            ref_path.write_text(original, encoding="utf-8")

    def test_list_verified_cases_raises_when_reference_file_missing(self) -> None:
        ref_path = PROJECT_ROOT / "references" / "reference_transcripts.json"
        backup = ref_path.read_bytes()
        try:
            ref_path.unlink()
            with self.assertRaises(FileNotFoundError):
                list_verified_cases()
        finally:
            ref_path.write_bytes(backup)


if __name__ == "__main__":
    unittest.main()
