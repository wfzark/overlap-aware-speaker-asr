from __future__ import annotations

import unittest

import json
import tempfile
from pathlib import Path

from src.compare_mixed_vs_separated import find_case, preview, read_existing_rows, read_json, select_cases, upsert_row
from src.config import get_audio_cases, load_config


class CompareMixedVsSeparatedHelpersTest(unittest.TestCase):
    def test_preview_collapses_whitespace_and_truncates(self) -> None:
        text = "你好   世界\n测试"
        self.assertEqual(preview(text, limit=10), "你好 世界 测试"[:10])

    def test_upsert_row_replaces_matching_case_and_model(self) -> None:
        rows = [{"case_id": "A", "model": "whisper-base", "mixed_text_length": 10}]
        updated = upsert_row(rows, {"case_id": "A", "model": "whisper-base", "mixed_text_length": 20})
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["mixed_text_length"], 20)

    def test_upsert_row_appends_new_case(self) -> None:
        rows = [{"case_id": "A", "model": "whisper-base"}]
        updated = upsert_row(rows, {"case_id": "B", "model": "whisper-base"})
        self.assertEqual(len(updated), 2)

    def test_find_case_returns_matching_case(self) -> None:
        config = load_config()
        case = find_case(config, "NoOverlap")
        self.assertEqual(case["id"], "NoOverlap")

    def test_find_case_raises_for_unknown_id(self) -> None:
        config = load_config()
        with self.assertRaises(ValueError):
            find_case(config, "__missing_case__")

    def test_select_cases_returns_all_when_requested(self) -> None:
        config = load_config()
        cases = select_cases(config, "all")
        self.assertEqual(len(cases), len(get_audio_cases(config)))

    def test_read_existing_rows_returns_empty_for_missing_csv(self) -> None:
        missing = Path("/tmp/__missing_compare_rows__.csv")
        self.assertEqual(read_existing_rows(missing), [])

    def test_read_json_loads_payload_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "payload.json"
            json_path.write_text(json.dumps({"text": "demo"}), encoding="utf-8")
            payload = read_json(json_path)
            self.assertEqual(payload["text"], "demo")


if __name__ == "__main__":
    unittest.main()
