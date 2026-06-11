from __future__ import annotations

import unittest

from src.separation_phase_diagram import (
    aggregate_trend_rows,
    build_gold_points,
    build_silver_points,
    compute_delta_cer,
    overlap_bin_key,
)


class SeparationPhaseDiagramTest(unittest.TestCase):
    def test_compute_delta_cer_is_separated_minus_mixed(self) -> None:
        self.assertEqual(compute_delta_cer(0.2, 0.5), 0.3)
        self.assertEqual(compute_delta_cer(0.5, 0.1), -0.4)

    def test_overlap_bin_key_rounds_to_step(self) -> None:
        self.assertEqual(overlap_bin_key(0.123), 0.1)
        self.assertEqual(overlap_bin_key(0.176), 0.2)

    def test_build_gold_points_marks_separation_help(self) -> None:
        rows = [
            {"case_id": "LightOverlap", "method": "mixed_whisper", "cer": "0.21"},
            {"case_id": "LightOverlap", "method": "separated_whisper", "cer": "0.48"},
            {"case_id": "LightOverlap", "method": "separated_whisper_cleaned", "cer": "0.38"},
            {"case_id": "NoOverlap", "method": "mixed_whisper", "cer": "0.22"},
            {"case_id": "NoOverlap", "method": "separated_whisper", "cer": "0.05"},
        ]
        points = build_gold_points(rows)
        by_id = {row["point_id"]: row for row in points}
        self.assertFalse(by_id["LightOverlap"]["separation_helps"])
        self.assertTrue(by_id["NoOverlap"]["separation_helps"])
        self.assertEqual(by_id["NoOverlap"]["source_label"], "stable/gold")
        self.assertEqual(by_id["NoOverlap"]["overlap_ratio_kind"], "tier_anchor")

    def test_build_silver_points_uses_manifest_overlap_ratio(self) -> None:
        cer_rows = [
            {
                "sample_id": "SyntheticLightOverlap_01",
                "tier": "SyntheticLightOverlap",
                "method": "mixed_whisper",
                "cer": "0.1",
            },
            {
                "sample_id": "SyntheticLightOverlap_01",
                "tier": "SyntheticLightOverlap",
                "method": "separated_whisper",
                "cer": "0.2",
            },
        ]
        manifest_rows = [
            {
                "sample_id": "SyntheticLightOverlap_01",
                "overlap_ratio": "0.15",
            }
        ]
        points = build_silver_points(cer_rows, manifest_rows, "synthetic/silver")
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["overlap_ratio"], 0.15)
        self.assertEqual(points[0]["overlap_ratio_kind"], "measured")

    def test_aggregate_trend_rows_computes_help_rate(self) -> None:
        points = [
            {"overlap_ratio": 0.11, "delta_cer_separated": -0.1, "separation_helps": True, "source_label": "synthetic/silver"},
            {"overlap_ratio": 0.12, "delta_cer_separated": 0.2, "separation_helps": False, "source_label": "synthetic/silver"},
        ]
        trend = aggregate_trend_rows(points, step=0.05)
        self.assertEqual(len(trend), 1)
        self.assertEqual(trend[0]["sample_count"], 2)
        self.assertEqual(trend[0]["separation_help_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
