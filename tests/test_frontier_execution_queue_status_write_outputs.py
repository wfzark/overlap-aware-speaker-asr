from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.frontier_execution_queue_status import (
    STATUS_COLUMNS,
    build_status_row,
    write_outputs,
)


class FrontierExecutionQueueStatusWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_status_csv_json_and_markdown(self) -> None:
        status_row = build_status_row(
            {"execution_chain_status": "execution_chain_ready"},
            {"execution_chain_status": "execution_chain_in_progress"},
            {"execution_chain_status": "execution_chain_ready"},
            {"overall_state": "receipt_ready_to_fill"},
            {"overall_state": "narrow_execution_ready"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.frontier_execution_queue_status.PROJECT_ROOT", root):
                csv_path, json_path, md_path = write_outputs(status_row)

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, STATUS_COLUMNS)
                self.assertEqual(list(reader)[0]["combined_chain_status"], "execution_chain_in_progress")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scope"], "frontier_execution_queues")
            self.assertIn("Frontier Execution Queue Status", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
