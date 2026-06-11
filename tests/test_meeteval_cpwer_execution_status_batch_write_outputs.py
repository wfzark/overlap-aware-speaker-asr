from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_execution_status import STATUS_COLUMNS
from src.meeteval_cpwer_execution_status_batch import build_status_rows, write_outputs


class MeetEvalCpwerExecutionStatusBatchWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_batch_status_artifacts(self) -> None:
        preflight_rows = [
            {"case_id": "NoOverlap", "preflight_pass": True},
            {"case_id": "HeavyOverlap", "preflight_pass": False},
        ]
        scaffold_rows = [
            {"case_id": "NoOverlap", "scaffold_status": "receipt_batch_scaffold_only"},
            {"case_id": "HeavyOverlap", "scaffold_status": "receipt_batch_scaffold_only"},
        ]
        receipt_by_case = {"NoOverlap": "template_only", "HeavyOverlap": "missing"}
        status_rows = build_status_rows(preflight_rows, scaffold_rows, receipt_by_case)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.meeteval_cpwer_execution_status_batch.PROJECT_ROOT", root):
                csv_path, json_path, md_path = write_outputs(status_rows)

            for path in (csv_path, json_path, md_path):
                self.assertTrue(path.exists())
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, STATUS_COLUMNS)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["execution_chain_status"], "execution_chain_ready")
                self.assertEqual(rows[1]["execution_chain_status"], "execution_chain_in_progress")
            self.assertIn("Status Batch", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
