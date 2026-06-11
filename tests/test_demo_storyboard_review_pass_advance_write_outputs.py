from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.demo_storyboard_review_pass import (
    REVIEW_COLUMNS,
    build_review_receipt_rows,
    build_review_row,
)
from src.demo_storyboard_review_pass_advance import (
    ADVANCE_COLUMNS,
    build_advance_row,
    write_outputs,
)


class DemoStoryboardReviewPassAdvanceWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_advance_second_review_and_receipt_artifacts(self) -> None:
        next_card = {"title": "Pipeline", "body": "ASR pipeline overview."}
        review_row = build_review_row(next_card, card_index=2)
        advance_row = build_advance_row(next_card, 2, "Problem")
        receipt_rows = build_review_receipt_rows(review_row, card_count=6)
        for receipt in receipt_rows:
            receipt["execution_status"] = "review_complete"
            receipt["review_scope"] = "second_storyboard_card"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.demo_storyboard_review_pass_advance.PROJECT_ROOT", root):
                paths = write_outputs(advance_row, review_row, receipt_rows)

            self.assertEqual(len(paths), 8)
            for path in paths:
                self.assertTrue(path.exists())
            with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ADVANCE_COLUMNS)
                self.assertEqual(list(reader)[0]["card_title"], "Pipeline")
            advance_payload = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(advance_payload["review_order"], "2")
            self.assertIn("Review Pass Advance", paths[2].read_text(encoding="utf-8"))
            with paths[3].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, REVIEW_COLUMNS)
                self.assertEqual(list(reader)[0]["review_order"], "2")
            receipt_payload = json.loads(paths[6].read_text(encoding="utf-8"))
            self.assertEqual(receipt_payload[0]["execution_status"], "review_complete")


if __name__ == "__main__":
    unittest.main()
