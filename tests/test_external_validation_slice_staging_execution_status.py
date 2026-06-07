from __future__ import annotations

import unittest

from src.external_validation_slice_staging_execution_status import build_status_row


class ExternalValidationSliceStagingExecutionStatusTest(unittest.TestCase):
    def test_build_status_row_marks_chain_ready_when_scaffold_complete(self) -> None:
        row = build_status_row(
            {"dataset_name": "AISHELL-4", "handoff_status": "staging_handoff_ready", "blocker": "license_confirmation_pending"},
            {"dataset_name": "AISHELL-4", "scaffold_status": "receipt_scaffold_only"},
            "template_only",
        )

        self.assertEqual(row["execution_chain_status"], "execution_chain_ready")

    def test_build_status_row_marks_chain_in_progress_when_scaffold_missing(self) -> None:
        row = build_status_row(
            {"dataset_name": "AISHELL-4"},
            {"scaffold_status": "missing"},
            "missing",
        )

        self.assertEqual(row["execution_chain_status"], "execution_chain_in_progress")


if __name__ == "__main__":
    unittest.main()
