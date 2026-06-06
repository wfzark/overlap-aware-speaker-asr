from __future__ import annotations

import unittest

from src.external_validation_candidates import (
    build_external_validation_candidate_lines,
    build_external_validation_candidate_rows,
    build_external_validation_prioritization_lines,
    build_external_validation_prioritization_rows,
)


class ExternalValidationCandidatesTest(unittest.TestCase):
    def test_build_external_validation_candidate_rows_cover_named_datasets(self) -> None:
        rows = build_external_validation_candidate_rows()

        by_name = {row["dataset_name"]: row for row in rows}

        self.assertIn("AISHELL-4", by_name)
        self.assertIn("AliMeeting", by_name)
        self.assertIn("AMI", by_name)
        self.assertIn("LibriCSS", by_name)
        self.assertEqual(by_name["AISHELL-4"]["label"], "external/sanity-check")
        self.assertIn("license", by_name["AISHELL-4"]["license_note"].lower())
        self.assertIn("overlap", by_name["LibriCSS"]["fit_note"].lower())

    def test_build_external_validation_candidate_lines_render_summary(self) -> None:
        lines = build_external_validation_candidate_lines(
            [
                {
                    "dataset_name": "AISHELL-4",
                    "label": "external/sanity-check",
                    "source_note": "Official AISHELL-4 release page.",
                    "license_note": "Check the official license before use.",
                    "fit_note": "Multi-speaker meeting data with realistic overlap.",
                    "first_preprocessing_step": "Map one small subset into the repository speaker-reference format.",
                    "next_action": "Confirm license and select a tiny sanity-check subset.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# External Validation Candidates", rendered)
        self.assertIn("AISHELL-4", rendered)
        self.assertIn("external/sanity-check", rendered)
        self.assertIn("Confirm license", rendered)

    def test_build_external_validation_prioritization_rows_recommend_first_dataset(self) -> None:
        rows = build_external_validation_prioritization_rows(build_external_validation_candidate_rows())

        by_name = {row["dataset_name"]: row for row in rows}

        self.assertEqual(by_name["AISHELL-4"]["priority_tier"], "start_here")
        self.assertEqual(by_name["AISHELL-4"]["recommended_order"], "1")
        self.assertIn("Chinese", by_name["AISHELL-4"]["why_now"])
        self.assertIn("license", by_name["AMI"]["readiness_note"].lower())
        self.assertEqual(by_name["LibriCSS"]["priority_tier"], "specialized_followup")

    def test_build_external_validation_prioritization_lines_render_ordered_card(self) -> None:
        lines = build_external_validation_prioritization_lines(
            [
                {
                    "dataset_name": "AISHELL-4",
                    "label": "external/sanity-check",
                    "priority_tier": "start_here",
                    "recommended_order": "1",
                    "readiness_note": "License check plus small-format mapping are still required.",
                    "why_now": "Chinese meeting overlap makes this the closest first sanity-check target.",
                    "next_action": "Confirm license and stage one tiny meeting slice.",
                }
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("# External Validation Prioritization", rendered)
        self.assertIn("start_here", rendered)
        self.assertIn("closest first sanity-check target", rendered)
        self.assertIn("recommended_order", rendered)


if __name__ == "__main__":
    unittest.main()
