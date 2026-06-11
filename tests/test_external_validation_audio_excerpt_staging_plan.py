from __future__ import annotations

import unittest

from src.external_validation_audio_excerpt_staging_plan import (
    apply_staging_plan_to_mapping,
    build_plan_row,
    build_reference_template,
)


class ExternalValidationAudioExcerptStagingPlanTest(unittest.TestCase):
    def test_build_reference_template_is_empty_segments(self) -> None:
        template = build_reference_template(
            {"slice_id": "stub", "dataset_name": "AISHELL-4", "label": "external/sanity-check"}
        )
        self.assertEqual(template["segments"], [])
        self.assertEqual(template["staging_status"], "reference_template_only")

    def test_apply_staging_plan_updates_mapping_status(self) -> None:
        updated = apply_staging_plan_to_mapping({"license_status": "confirmed_research_cc_by_sa_4_0"})
        self.assertEqual(updated["staging_status"], "awaiting_local_audio_download")
        self.assertEqual(updated["mapping_status"], "reference_template_staged")

    def test_build_plan_row_reports_reference_staged(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio.wav"
            reference = root / "ref.json"
            reference.write_text("{}", encoding="utf-8")
            row = build_plan_row(
                {
                    "dataset_name": "AISHELL-4",
                    "slice_id": "stub",
                    "label": "external/sanity-check",
                    "license_status": "confirmed",
                    "audio_path": "audio.wav",
                    "reference_path": "ref.json",
                    "staging_status": "awaiting_local_audio_download",
                },
                audio,
                reference,
            )
            self.assertEqual(row["reference_template_staged"], "True")
            self.assertEqual(row["audio_staged"], "False")


if __name__ == "__main__":
    unittest.main()
