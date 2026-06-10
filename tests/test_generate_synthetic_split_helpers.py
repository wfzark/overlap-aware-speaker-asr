from __future__ import annotations

import unittest
from pathlib import Path

from src.generate_synthetic_split import (
    choose_pair,
    sample_index_within_split,
    split_for_index,
    tier_gap_sec,
    tier_overlap_ratio,
)


class GenerateSyntheticSplitHelpersTest(unittest.TestCase):
    def test_split_for_index_assigns_dev_then_test(self) -> None:
        self.assertEqual(split_for_index(0), "dev")
        self.assertEqual(split_for_index(9), "dev")
        self.assertEqual(split_for_index(10), "test")

    def test_sample_index_within_split_wraps_every_ten(self) -> None:
        self.assertEqual(sample_index_within_split(0), 0)
        self.assertEqual(sample_index_within_split(9), 9)
        self.assertEqual(sample_index_within_split(10), 0)

    def test_choose_pair_uses_index_modulo_for_con_files(self) -> None:
        con_files = [Path("con_a.wav"), Path("con_b.wav")]
        pro_files = [Path("pro_x.wav"), Path("pro_y.wav")]
        self.assertEqual(choose_pair(con_files, pro_files, 2), (con_files[0], pro_files[0]))

    def test_tier_overlap_ratio_scales_with_split_sample_index(self) -> None:
        first = tier_overlap_ratio("SyntheticMidOverlap", 0)
        later = tier_overlap_ratio("SyntheticMidOverlap", 19)
        self.assertLess(first, later)

    def test_tier_gap_sec_returns_zero_for_overlap_only_tiers(self) -> None:
        self.assertEqual(tier_gap_sec("SyntheticHeavyOverlap", 3), 0.0)


if __name__ == "__main__":
    unittest.main()
