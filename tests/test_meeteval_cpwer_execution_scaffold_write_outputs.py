from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.meeteval_cpwer_execution_scaffold import (
    SCAFFOLD_COLUMNS,
    build_scaffold_receipt_rows,
    build_scaffold_row,
    write_outputs,
)


class MeetEvalCpwerExecutionScaffoldWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_scaffold_and_receipt_artifacts(self) -> None:
        scaffold_row = build_scaffold_row(
            {"case_id": "NoOverlap", "bridge_status": "cpwer_bridge_complete", "cpwer_bridge_lite": "0.12"}
        )
        receipt_rows = build_scaffold_receipt_rows(scaffold_row)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.meeteval_cpwer_execution_scaffold.PROJECT_ROOT", root):
                outputs = write_outputs(scaffold_row, receipt_rows)

            for path in outputs:
                self.assertTrue(path.exists())
            with outputs[0].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCAFFOLD_COLUMNS)
                self.assertEqual(list(reader)[0]["scaffold_status"], "scaffold_only")
            self.assertIn("Execution Scaffold", outputs[2].read_text(encoding="utf-8"))
            self.assertEqual(json.loads(outputs[3].read_text(encoding="utf-8"))[0]["execution_status"], "scaffold_complete")


if __name__ == "__main__":
    unittest.main()
