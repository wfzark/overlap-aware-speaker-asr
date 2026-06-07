from __future__ import annotations

import unittest

from src.external_validation_slice_manifest import (
    build_manifest_lines,
    build_manifest_receipt_lines,
    build_manifest_receipt_rows,
    build_manifest_row,
)


class ExternalValidationSliceManifestTest(unittest.TestCase):
    def test_build_manifest_row_blocks_staging_when_license_pending(self) -> None:
        row = build_manifest_row(
            {
                "dataset_name": "AISHELL-4",
                "slice_id": "aishell4_meeting_excerpt_stub_001",
                "label": "external/sanity-check",
                "license_status": "pending_confirmation",
                "mapping_status": "scaffold_only",
                "audio_path": "resources/external_sanity_check/aishell4/meeting_excerpt_stub_001.wav",
                "reference_path": "resources/external_sanity_check/aishell4/meeting_excerpt_stub_001_reference.json",
            }
        )

        self.assertEqual(row["staging_status"], "blocked_by_license_gate")

    def test_build_manifest_lines_render_manifest(self) -> None:
        lines = build_manifest_lines(
            {
                "dataset_name": "AISHELL-4",
                "slice_id": "aishell4_meeting_excerpt_stub_001",
                "label": "external/sanity-check",
                "license_status": "pending_confirmation",
                "mapping_status": "scaffold_only",
                "audio_path": "audio.wav",
                "reference_path": "reference.json",
                "staging_status": "blocked_by_license_gate",
                "manifest_note": "Manifest-only staging plan.",
            }
        )
        rendered = "\n".join(lines)

        self.assertIn("# External Validation Slice Manifest", rendered)
        self.assertIn("blocked_by_license_gate", rendered)

    def test_build_manifest_receipt_rows_mark_manifest_complete(self) -> None:
        rows = build_manifest_receipt_rows({"dataset_name": "AISHELL-4", "staging_status": "blocked_by_license_gate"})

        self.assertEqual(rows[0]["execution_status"], "manifest_complete")
        self.assertIn("blocked", rows[0]["writeback_note"].lower())

    def test_build_manifest_receipt_lines_render_receipt(self) -> None:
        lines = build_manifest_receipt_lines(
            [
                {
                    "execution_status": "manifest_complete",
                    "slice_scope": "single_short_meeting_excerpt",
                    "dataset_name": "AISHELL-4",
                    "staging_status": "blocked_by_license_gate",
                    "writeback_note": "Slice manifest documented; external audio staging remains blocked until license confirmation is recorded.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("manifest_complete", rendered)


if __name__ == "__main__":
    unittest.main()
