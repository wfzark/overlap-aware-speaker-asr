from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.frontier_execution_receipt_fill_execution_completion_summary import (
    COMPLETION_COLUMNS,
    build_completion_summary_row,
    write_outputs,
)


class FrontierExecutionReceiptFillExecutionCompletionSummaryWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_csv_json_and_markdown(self) -> None:
        completion_row = build_completion_summary_row(
            {
                "meeteval_fill_execution_status": "awaiting_fill",
                "speaker_profile_fill_execution_status": "fill_complete",
                "external_staging_fill_execution_status": "awaiting_fill",
                "combined_fill_execution_status": "fill_execution_in_progress",
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.frontier_execution_receipt_fill_execution_completion_summary.PROJECT_ROOT",
                root,
            ):
                csv_path, json_path, md_path = write_outputs(completion_row)

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, COMPLETION_COLUMNS)
                self.assertEqual(list(reader)[0]["awaiting_fill_execution_count"], "2")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["fill_execution_complete_count"], "1")
            self.assertIn(
                "Fill Execution Completion Summary",
                md_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
