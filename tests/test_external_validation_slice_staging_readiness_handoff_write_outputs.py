from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.external_validation_slice_staging_readiness_handoff import (
    HANDOFF_COLUMNS,
    build_handoff_receipt_rows,
    build_handoff_row,
    write_outputs,
)


class ExternalValidationSliceStagingReadinessHandoffWriteOutputsTest(unittest.TestCase):
    def test_write_outputs_emits_handoff_and_receipt_artifacts(self) -> None:
        handoff_row = build_handoff_row(
            {
                "dataset_name": "FixtureDataset",
                "readiness_status": "not_ready",
                "blocker": "license_confirmation_pending",
            }
        )
        receipt_rows = build_handoff_receipt_rows(handoff_row)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("src.external_validation_slice_staging_readiness_handoff.PROJECT_ROOT", root):
                outputs = write_outputs(handoff_row, receipt_rows)

            for path in outputs:
                self.assertTrue(path.exists(), msg=str(path))

            csv_path = outputs[0]
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, HANDOFF_COLUMNS)
                self.assertEqual(list(reader)[0]["dataset_name"], "FixtureDataset")

            receipt_payload = json.loads(outputs[3].read_text(encoding="utf-8"))
            self.assertEqual(receipt_payload[0]["execution_status"], "handoff_documented")


if __name__ == "__main__":
    unittest.main()
