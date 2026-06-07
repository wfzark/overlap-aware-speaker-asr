from __future__ import annotations

import unittest

from src.external_validation_slice_staging_execution_status_bridge_checklist import build_bridge_checklist_rows


class ExternalValidationSliceStagingExecutionStatusBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_uses_chain_status(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "dataset_name": "AISHELL-4",
                "execution_chain_status": "execution_chain_ready",
                "blocker": "license_confirmation_pending",
            }
        )

        self.assertEqual(rows[0]["execution_chain_status"], "execution_chain_ready")
        self.assertIn("AISHELL-4", rows[0]["checklist_goal"])

    def test_build_bridge_checklist_rows_defaults_dataset_name(self) -> None:
        rows = build_bridge_checklist_rows({})

        self.assertEqual(rows[0]["dataset_name"], "AISHELL-4")


if __name__ == "__main__":
    unittest.main()
