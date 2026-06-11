from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_execution_receipt_scaffold import (
    SCAFFOLD_COLUMNS,
    build_scaffold_receipt_rows,
    build_scaffold_row,
    write_outputs,
)


class MeetEvalCpwerExecutionReceiptScaffoldWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_scaffold_receipt_template_and_receipt_artifacts(self) -> None:
        scaffold_row = build_scaffold_row(
            {"case_id": "NoOverlap", "preflight_pass": True, "hypothesis_source": "separated_whisper"}
        )
        receipt_rows = build_scaffold_receipt_rows(scaffold_row)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.meeteval_cpwer_execution_receipt_scaffold.PROJECT_ROOT", root):
                outputs = write_outputs(scaffold_row, receipt_rows)

            for path in outputs:
                self.assertTrue(path.exists())
            with outputs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCAFFOLD_COLUMNS)
                self.assertEqual(list(reader)[0]["scaffold_status"], "receipt_scaffold_only")
            self.assertEqual(
                json.loads(outputs[3].read_text(encoding="utf-8"))[0]["execution_status"],
                "template_only",
            )
            self.assertEqual(
                json.loads(outputs[4].read_text(encoding="utf-8"))[0]["execution_status"],
                "receipt_scaffold_complete",
            )


if __name__ == "__main__":
    unittest.main()
