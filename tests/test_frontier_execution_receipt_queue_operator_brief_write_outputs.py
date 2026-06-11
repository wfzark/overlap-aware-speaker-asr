from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.frontier_execution_receipt_queue_operator_brief import (
    OPERATOR_BRIEF_COLUMNS,
    build_operator_brief_row,
    write_outputs,
)


class FrontierExecutionReceiptQueueOperatorBriefWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_csv_json_and_markdown(self) -> None:
        row = build_operator_brief_row(
            {
                "queue_status": "queue_in_progress",
                "ready_receipt_count": "2",
                "pending_receipt_count": "1",
            },
            [
                {
                    "frontier_name": "meeteval_compatibility",
                    "readiness_status": "receipt_ready_to_fill",
                    "recommended_action": "Update execution receipt after real run.",
                    "expected_outputs": "results/tables/meeteval_cpwer_execution_receipt.json",
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "src.frontier_execution_receipt_queue_operator_brief.PROJECT_ROOT",
                root,
            ):
                csv_path, json_path, md_path = write_outputs(row)

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, OPERATOR_BRIEF_COLUMNS)
                self.assertEqual(list(reader)[0]["operator_frontier"], "meeteval_compatibility")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("ready_receipt_count=2", payload["operator_urgency"])
            self.assertIn("Operator Brief", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
