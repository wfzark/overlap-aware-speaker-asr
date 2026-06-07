from __future__ import annotations

import unittest

from src.external_validation_license_gate_bridge_checklist import (
    build_bridge_checklist_lines,
    build_bridge_checklist_rows,
)


class ExternalValidationLicenseGateBridgeChecklistTest(unittest.TestCase):
    def test_build_bridge_checklist_rows_use_gate_row(self) -> None:
        rows = build_bridge_checklist_rows(
            {
                "dataset_name": "AISHELL-4",
                "license_status": "pending_confirmation",
            }
        )

        self.assertEqual(rows[0]["dataset_name"], "AISHELL-4")
        self.assertIn("pending_confirmation", rows[0]["bridge_note"])

    def test_build_bridge_checklist_lines_render_note(self) -> None:
        lines = build_bridge_checklist_lines(
            [
                {
                    "checklist_order": "1",
                    "dataset_name": "AISHELL-4",
                    "license_status": "pending_confirmation",
                    "prerequisite_artifact": "results/figures/external_validation_license_gate.md",
                    "receipt_target": "results/figures/external_validation_slice_manifest.md",
                    "checklist_goal": "Verify the license gate bridge.",
                    "bridge_note": "License status remains pending_confirmation.",
                    "next_gate": "Confirm this bridge before opening the external slice manifest target.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# External Validation License Gate Bridge Checklist", rendered)


if __name__ == "__main__":
    unittest.main()
