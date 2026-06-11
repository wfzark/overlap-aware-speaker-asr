from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.external_validation_slice_staging_handoff_receipt_readiness import (
    READINESS_COLUMNS,
    build_readiness_row,
    write_outputs,
)


class ExternalValidationSliceStagingHandoffReceiptReadinessWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_csv_json_and_markdown(self) -> None:
        row = build_readiness_row(
            {"dataset_name": "FixtureDataset", "execution_chain_status": "execution_chain_ready"},
            {"execution_status": "template_only"},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.external_validation_slice_staging_handoff_receipt_readiness.PROJECT_ROOT", root):
                csv_path, json_path, md_path = write_outputs(row)

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, READINESS_COLUMNS)
                self.assertEqual(list(reader)[0]["readiness_status"], "receipt_ready_to_fill")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_name"], "FixtureDataset")
            self.assertIn("Handoff Receipt Readiness", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
