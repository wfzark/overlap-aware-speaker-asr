from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.external_validation_slice_staging_execution_status import (
    STATUS_COLUMNS,
    build_status_row,
    write_outputs,
)


class ExternalValidationSliceStagingExecutionStatusBuildRowTest(unittest.TestCase):
    def test_build_status_row_marks_chain_ready_when_scaffold_only_and_blocker_present(self) -> None:
        row = build_status_row(
            {"dataset_name": "FixtureDataset", "handoff_status": "staging_handoff_ready", "blocker": "license"},
            {"scaffold_status": "receipt_scaffold_only"},
            "scaffold_only",
        )
        self.assertEqual(row["dataset_name"], "FixtureDataset")
        self.assertEqual(row["execution_chain_status"], "execution_chain_ready")

    def test_build_status_row_marks_in_progress_when_scaffold_missing(self) -> None:
        row = build_status_row(
            {"dataset_name": "FixtureDataset", "blocker": "license"},
            {"scaffold_status": "missing"},
            "missing",
        )
        self.assertEqual(row["execution_chain_status"], "execution_chain_in_progress")


if __name__ == "__main__":
    unittest.main()
