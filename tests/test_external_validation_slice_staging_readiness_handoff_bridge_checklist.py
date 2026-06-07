from __future__ import annotations

import unittest

from src.external_validation_slice_staging_readiness_handoff_bridge_checklist import build_bridge_checklist_rows


class ExternalValidationSliceStagingReadinessHandoffBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_targets_staging_receipt(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "dataset_name": "AISHELL-4",
                "handoff_status": "staging_handoff_ready",
                "blocker": "license_confirmation_pending",
            }
        )

        self.assertEqual(rows[0]["dataset_name"], "AISHELL-4")
        self.assertIn("staging_handoff_receipt", rows[0]["receipt_target"])


if __name__ == "__main__":
    unittest.main()
