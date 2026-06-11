from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.external_validation_slice_staging_handoff_receipt_scaffold import (
    SCAFFOLD_COLUMNS,
    build_scaffold_receipt_rows,
    build_scaffold_row,
    write_outputs,
)


class ExternalValidationSliceStagingHandoffReceiptScaffoldWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_scaffold_and_receipt_artifacts(self) -> None:
        scaffold_row = build_scaffold_row(
            {
                "dataset_name": "FixtureDataset",
                "handoff_status": "staging_handoff_ready",
                "blocker": "license",
            }
        )
        receipt_rows = build_scaffold_receipt_rows(scaffold_row)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.external_validation_slice_staging_handoff_receipt_scaffold.PROJECT_ROOT", root):
                outputs = write_outputs(scaffold_row, receipt_rows)

            for path in outputs:
                self.assertTrue(path.exists(), msg=str(path))

            csv_path = outputs[0]
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SCAFFOLD_COLUMNS)
                self.assertEqual(list(reader)[0]["dataset_name"], "FixtureDataset")

            receipt_template = json.loads(outputs[3].read_text(encoding="utf-8"))
            self.assertEqual(receipt_template[0]["execution_status"], "template_only")


if __name__ == "__main__":
    unittest.main()
