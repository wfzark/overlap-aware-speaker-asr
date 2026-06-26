"""Tests for RQ45: multi-meeting threshold stability simulation (experimental/frontier).

Pins the pure helpers: ``bootstrap_resample``, ``calibrate_threshold_at_spec``,
``count_modes``, ``percentile_interval_width``. Also pins the lang-id entropy
detector primitives (RQ13/RQ16/RQ25/RQ44 verbatim), the ``expected_oob_size``
helper, the ``out_of_bag_cpwer`` helper, module constants, and a smoke test
that the in-sample calibration on the real 77-window AISHELL-4 data reproduces
RQ25/RQ44's threshold (0.38, sensitivity 35/37, specificity 37/40), plus a
smoke test that the n=77 bootstrap median threshold is 0.38 (RQ44 reproduction).

No Whisper / no audio needed. numpy + stdlib only. unittest (not pytest).
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# The RQ45 analysis script lives in results/frontier/ as a standalone module
# (no src. package). Import it via sys.path manipulation, mirroring the
# harness test_bootstrap_threshold pattern.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _PROJECT_ROOT / "results" / "frontier" / "multi_meeting_threshold"
sys.path.insert(0, str(_SCRIPT_DIR))

import multi_meeting_threshold_analysis as rq45  # noqa: E402  (path-injected import)

AISHELL4_JSON = (
    _PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)


# ------------------------------------------------------------- module constants
class TestModuleConstants(unittest.TestCase):
    def test_sample_sizes(self) -> None:
        self.assertEqual(rq45.SAMPLE_SIZES, [77, 154, 308, 616, 1232])

    def test_sample_sizes_are_multiples_of_77(self) -> None:
        for n in rq45.SAMPLE_SIZES:
            self.assertEqual(n % 77, 0)
            self.assertGreaterEqual(n, 77)

    def test_anchor_n_is_616(self) -> None:
        # H45a/b/c are all anchored at n=616 (8x the original 77).
        self.assertEqual(rq45.H45_ANCHOR_N, 616)
        self.assertIn(rq45.H45_ANCHOR_N, rq45.SAMPLE_SIZES)

    def test_n_boot_is_2000(self) -> None:
        self.assertEqual(rq45.N_BOOT, 2000)

    def test_seed_is_42(self) -> None:
        self.assertEqual(rq45.SEED, 42)

    def test_mode_min_fraction_is_5pct(self) -> None:
        self.assertEqual(rq45.MODE_MIN_FRACTION, 0.05)

    def test_hypothesis_kill_thresholds(self) -> None:
        self.assertEqual(rq45.H45A_MAX_MODES, 2)
        self.assertEqual(rq45.H45B_MAX_WIDTH, 0.20)
        self.assertEqual(rq45.H45C_MAX_CPWER, 1.05)

    def test_threshold_grid(self) -> None:
        self.assertEqual(rq45.THRESHOLD_GRID[0], 0.00)
        self.assertEqual(rq45.THRESHOLD_GRID[-1], 2.00)
        self.assertEqual(len(rq45.THRESHOLD_GRID), 201)
        self.assertEqual(rq45.THRESHOLD_GRID[38], 0.38)


# ------------------------------------------------------------- bootstrap_resample
class TestBootstrapResample(unittest.TestCase):
    def test_shape_n_equals_population(self) -> None:
        idx = rq45.bootstrap_resample(77, 77, 100, 42)
        self.assertEqual(idx.shape, (100, 77))

    def test_shape_n_sample_exceeds_population(self) -> None:
        # RQ45 simulates larger corpora: n_sample may exceed n_population.
        idx = rq45.bootstrap_resample(77, 616, 50, 42)
        self.assertEqual(idx.shape, (50, 616))

    def test_values_in_range(self) -> None:
        idx = rq45.bootstrap_resample(77, 616, 10, 42)
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 77))

    def test_deterministic_given_seed(self) -> None:
        a = rq45.bootstrap_resample(77, 154, 5, 42)
        b = rq45.bootstrap_resample(77, 154, 5, 42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = rq45.bootstrap_resample(77, 154, 5, 42)
        b = rq45.bootstrap_resample(77, 154, 5, 7)
        self.assertFalse(np.array_equal(a, b))

    def test_duplicates_allowed_when_n_sample_exceeds_population(self) -> None:
        # With n_sample=1232 draws from 77 windows, every window must appear at
        # least once with overwhelming probability (P(window absent) =
        # (76/77)^1232 ~ 1e-7). This is the structural fact that makes the OOB
        # set empty at large n.
        idx = rq45.bootstrap_resample(77, 1232, 1, 42)
        unique_count = len(np.unique(idx[0]))
        self.assertEqual(unique_count, 77)


# ------------------------------------------------------- calibrate_threshold_at_spec
class TestCalibrateThresholdAtSpec(unittest.TestCase):
    def test_separable_case(self) -> None:
        # negs in [0, 0.20], pos at 1.0/1.1 (within the 0..2.0 bit entropy
        # grid). A threshold just above 0.20 separates them: sens=1, spec=1.
        scores = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 1.0, 1.1])
        labels = np.array([0, 0, 0, 0, 0, 1, 1])
        out = rq45.calibrate_threshold_at_spec(scores, labels)
        self.assertGreaterEqual(out["specificity"], 0.90 - 1e-9)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["tp"], 2)
        self.assertEqual(out["fn"], 0)

    def test_target_specificity_respected(self) -> None:
        scores = np.array([0., 0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1])
        out = rq45.calibrate_threshold_at_spec(scores, labels, target_spec=0.75)
        self.assertGreaterEqual(out["specificity"], 0.75 - 1e-9)

    def test_tie_breaker_lower_threshold(self) -> None:
        # Among thresholds that achieve the same (sens, spec), the LOWER one
        # wins (more sensitive operating point). Here both 0.01 and 0.5 give
        # sens=1, spec=1 (negs at 0.0, pos at 0.5), so 0.01 is picked.
        scores = np.array([0.0, 0.0, 0.5, 0.5])
        labels = np.array([0, 0, 1, 1])
        out = rq45.calibrate_threshold_at_spec(scores, labels)
        self.assertEqual(out["threshold"], 0.01)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["specificity"], 1.0)

    def test_fallback_when_no_threshold_meets_specificity(self) -> None:
        # Impossible case: a positive at score 0.0 forces specificity < 0.90
        # at any threshold <= 0.0. The fallback is the highest grid threshold.
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        out = rq45.calibrate_threshold_at_spec(scores, labels, target_spec=0.90)
        # With a positive at 0.0, any threshold <= 0.0 has FP >= 1 (spec=0.8 < 0.9);
        # only thresholds > 0.0 give spec=1.0 but sens=0. The fallback picks the
        # highest threshold when no threshold satisfies spec AND has sens > 0;
        # here thresholds > 0 satisfy spec but have sens=0, so the first such
        # threshold (0.01) wins (max sens among spec>=0.9 is 0, tie-break lower).
        self.assertGreaterEqual(out["specificity"], 0.90 - 1e-9)
        self.assertEqual(out["sensitivity"], 0.0)

    def test_returns_dict_has_required_fields(self) -> None:
        scores = np.array([0.1, 0.9])
        labels = np.array([0, 1])
        out = rq45.calibrate_threshold_at_spec(scores, labels)
        for k in ("threshold", "sensitivity", "specificity", "tp", "fp", "tn", "fn"):
            self.assertIn(k, out)


# ------------------------------------------------------------- count_modes
class TestCountModes(unittest.TestCase):
    def test_unimodal(self) -> None:
        thr = np.array([0.38] * 95 + [0.87] * 5, dtype=float)
        out = rq45.count_modes(thr, min_fraction=0.05)
        # 0.38 has 95% (>= 5%), 0.87 has 5% (>= 5% with EPS tolerance) -> 2 modes.
        self.assertEqual(out["n_modes"], 2)
        self.assertEqual(out["dominant"], 0.38)
        self.assertAlmostEqual(out["dominant_fraction"], 0.95, places=6)

    def test_strictly_unimodal(self) -> None:
        thr = np.array([0.38] * 99 + [0.87] * 1, dtype=float)
        out = rq45.count_modes(thr, min_fraction=0.05)
        # Only 0.38 has >= 5% frequency; 0.87 has 1% (< 5%) -> 1 mode.
        self.assertEqual(out["n_modes"], 1)
        self.assertEqual(out["modes"][0]["threshold"], 0.38)

    def test_six_modal_counts_six(self) -> None:
        # Six thresholds each with >= 5% frequency -> 6 modes (this is the
        # RQ44 n=77 situation under the >=5% rule, except 0.84 was < 5%; here
        # we force all six to be modes).
        thr = np.array(
            [0.38] * 30 + [0.87] * 20 + [0.01] * 15 + [0.95] * 15 + [0.33] * 10
            + [0.84] * 10,
            dtype=float,
        )
        out = rq45.count_modes(thr, min_fraction=0.05)
        self.assertEqual(out["n_modes"], 6)
        self.assertEqual(out["n_unique"], 6)

    def test_empty_input(self) -> None:
        out = rq45.count_modes(np.array([], dtype=float), min_fraction=0.05)
        self.assertEqual(out["n_modes"], 0)
        self.assertTrue(math.isnan(out["dominant"]))
        self.assertEqual(out["dominant_fraction"], 0.0)

    def test_modes_sorted_by_descending_count(self) -> None:
        thr = np.array([0.38] * 50 + [0.87] * 30 + [0.01] * 20, dtype=float)
        out = rq45.count_modes(thr, min_fraction=0.05)
        counts = [m["count"] for m in out["modes"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(out["modes"][0]["threshold"], 0.38)

    def test_min_fraction_boundary(self) -> None:
        # Exactly at the 5% boundary (with EPS tolerance): should count as a mode.
        thr = np.array([0.38] * 95 + [0.87] * 5, dtype=float)
        out = rq45.count_modes(thr, min_fraction=0.05)
        self.assertEqual(out["n_modes"], 2)
        # Just below 5%: should NOT count.
        thr2 = np.array([0.38] * 96 + [0.87] * 4, dtype=float)
        out2 = rq45.count_modes(thr2, min_fraction=0.05)
        self.assertEqual(out2["n_modes"], 1)


# ------------------------------------------------------ percentile_interval_width
class TestPercentileIntervalWidth(unittest.TestCase):
    def test_width_basic(self) -> None:
        # 2.5th pct = 1.0, 97.5th pct = 10.0 for [1..10] -> width 9.0
        vals = np.arange(1, 11, dtype=float)
        w = rq45.percentile_interval_width(vals, 2.5, 97.5)
        self.assertGreater(w, 0.0)
        self.assertLess(w, 9.5)

    def test_zero_width_for_constant(self) -> None:
        vals = np.array([0.38] * 100, dtype=float)
        w = rq45.percentile_interval_width(vals, 2.5, 97.5)
        self.assertEqual(w, 0.0)

    def test_nan_filtered(self) -> None:
        # NaNs are filtered before computing percentiles.
        vals = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0], dtype=float)
        w = rq45.percentile_interval_width(vals, 2.5, 97.5)
        self.assertFalse(math.isnan(w))

    def test_empty_returns_nan(self) -> None:
        w = rq45.percentile_interval_width(np.array([], dtype=float), 2.5, 97.5)
        self.assertTrue(math.isnan(w))

    def test_all_nan_returns_nan(self) -> None:
        w = rq45.percentile_interval_width(
            np.array([np.nan, np.nan], dtype=float), 2.5, 97.5
        )
        self.assertTrue(math.isnan(w))

    def test_custom_lo_hi(self) -> None:
        vals = np.arange(1, 101, dtype=float)  # 1..100
        w = rq45.percentile_interval_width(vals, 25, 75)
        # 25th pct = 25.75, 75th pct = 75.25 (numpy linear interp) -> ~49.5
        self.assertAlmostEqual(w, 49.5, places=1)


# ------------------------------------------------------------- expected_oob_size
class TestExpectedOobSize(unittest.TestCase):
    def test_n_equals_population(self) -> None:
        # n=77, pop=77 -> expected OOB = 77 * (76/77)^77 ~ 28.14 (the classic
        # bootstrap OOB fraction ~ 1/e ~ 36.8%).
        s = rq45.expected_oob_size(77, 77)
        self.assertAlmostEqual(s, 28.14, delta=0.5)

    def test_large_n_shrinks_oob(self) -> None:
        # n=616, pop=77 -> expected OOB ~ 0.025 (essentially empty).
        s = rq45.expected_oob_size(77, 616)
        self.assertLess(s, 0.05)
        self.assertGreater(s, 0.0)

    def test_very_large_n_near_zero(self) -> None:
        s = rq45.expected_oob_size(77, 1232)
        self.assertLess(s, 1e-4)

    def test_zero_population(self) -> None:
        self.assertEqual(rq45.expected_oob_size(0, 10), 0.0)


# ------------------------------------------------------------- out_of_bag_cpwer
class TestOutOfBagCpwer(unittest.TestCase):
    def test_empty_oob_returns_nan(self) -> None:
        # If every window is in-bag, OOB is empty -> cpwer is nan.
        scores = np.array([0.1, 0.5, 0.9])
        mixed = np.array([1.0, 1.2, 1.3])
        sep = np.array([1.0, 1.5, 2.0])
        in_bag = np.array([0, 1, 2])  # all in-bag
        out = rq45.out_of_bag_cpwer(scores, mixed, sep, 0.4, in_bag)
        self.assertTrue(math.isnan(out["cpwer"]))
        self.assertEqual(out["n_oob"], 0)

    def test_oob_routing(self) -> None:
        # 4 windows; in-bag = {0, 1}; OOB = {2, 3}.
        # threshold = 0.5: window 2 (score 0.4 < 0.5) -> SEPARATED (1.5);
        #                  window 3 (score 0.9 >= 0.5) -> MIXED (1.3).
        scores = np.array([0.1, 0.2, 0.4, 0.9])
        mixed = np.array([1.0, 1.1, 1.2, 1.3])
        sep = np.array([1.0, 1.0, 1.5, 2.0])
        in_bag = np.array([0, 1])
        out = rq45.out_of_bag_cpwer(scores, mixed, sep, 0.5, in_bag)
        self.assertEqual(out["n_oob"], 2)
        self.assertEqual(out["n_flagged_mixed"], 1)
        self.assertEqual(out["n_separated"], 1)
        self.assertAlmostEqual(out["cpwer"], (1.5 + 1.3) / 2.0, places=6)

    def test_duplicates_in_in_bag_dont_double_count_oob(self) -> None:
        # in_bag with duplicates: {0, 0, 1, 1}; OOB should still be {2, 3}.
        scores = np.array([0.1, 0.2, 0.4, 0.9])
        mixed = np.array([1.0, 1.1, 1.2, 1.3])
        sep = np.array([1.0, 1.0, 1.5, 2.0])
        in_bag = np.array([0, 0, 1, 1])
        out = rq45.out_of_bag_cpwer(scores, mixed, sep, 0.5, in_bag)
        self.assertEqual(out["n_oob"], 2)


# ------------------------------------------------------------- detector primitives
class TestDetectorPrimitives(unittest.TestCase):
    def test_script_category_han(self) -> None:
        self.assertEqual(rq45.script_category("中"), "Han")

    def test_script_category_latin(self) -> None:
        self.assertEqual(rq45.script_category("A"), "Latin")

    def test_script_category_space(self) -> None:
        self.assertEqual(rq45.script_category(" "), "Space")

    def test_script_category_digit(self) -> None:
        self.assertEqual(rq45.script_category("7"), "Digit")

    def test_language_id_entropy_clean_chinese_is_low(self) -> None:
        # Near-monoscript Han -> entropy ~ 0.
        h = rq45.language_id_entropy("你好世界今天是星期天")
        self.assertLess(h, 0.05)

    def test_language_id_entropy_mixed_scripts_is_high(self) -> None:
        # Han + Latin + Katakana + Hangul -> high entropy.
        h = rq45.language_id_entropy("你好ABCカザ가")
        self.assertGreater(h, 1.0)

    def test_language_id_entropy_empty_is_zero(self) -> None:
        self.assertEqual(rq45.language_id_entropy(""), 0.0)
        self.assertEqual(rq45.language_id_entropy("   "), 0.0)

    def test_max_across_speakers(self) -> None:
        # MAX-aggregation: the worst-case (highest-entropy) speaker track wins.
        # s0 is monoscript Han (entropy ~0); s1 mixes Han + Latin (entropy > 0).
        # max(0, >0) = >0.
        window = {"separated_text_per_speaker": {"s0": "你好", "s1": "你好ABC"}}
        v = rq45.max_across_speakers(window)
        self.assertGreater(v, 0.0)

    def test_max_across_speakers_empty(self) -> None:
        window = {"separated_text_per_speaker": {}}
        self.assertEqual(rq45.max_across_speakers(window), 0.0)


# ---------------------------------------------------- in-sample + n=77 smoke tests
class TestInSampleReproduction(unittest.TestCase):
    """Smoke test: in-sample calibration on 77 windows reproduces RQ25/RQ44's 0.38."""

    @classmethod
    def setUpClass(cls) -> None:
        data = json.loads(AISHELL4_JSON.read_text(encoding="utf-8"))
        cls.windows = data["windows"]
        cls.lang_ent = np.array(
            [rq45.max_across_speakers(w) for w in cls.windows], dtype=float
        )
        cls.sep_cpwer = np.array(
            [float(w["always_separated_cpwer"]) for w in cls.windows], dtype=float
        )
        cls.mixed_cpwer = np.array(
            [float(w["always_mixed_cpwer"]) for w in cls.windows], dtype=float
        )
        cls.labels = (cls.sep_cpwer > rq45.CATASTROPHIC_CPWER).astype(int)

    def test_seventy_seven_windows(self) -> None:
        self.assertEqual(len(self.windows), 77)

    def test_label_counts_match_rq44(self) -> None:
        # RQ44: 37 hallucinated, 40 clean.
        self.assertEqual(int(self.labels.sum()), 37)
        self.assertEqual(int((self.labels == 0).sum()), 40)

    def test_in_sample_threshold_is_0_38(self) -> None:
        out = rq45.calibrate_threshold_at_spec(self.lang_ent, self.labels)
        self.assertAlmostEqual(out["threshold"], 0.38, places=2)

    def test_in_sample_sensitivity_is_35_of_37(self) -> None:
        out = rq45.calibrate_threshold_at_spec(self.lang_ent, self.labels)
        self.assertEqual(out["tp"], 35)
        self.assertEqual(out["fn"], 2)
        self.assertAlmostEqual(out["sensitivity"], 35 / 37, places=4)

    def test_in_sample_specificity_is_37_of_40(self) -> None:
        out = rq45.calibrate_threshold_at_spec(self.lang_ent, self.labels)
        self.assertEqual(out["tn"], 37)
        self.assertEqual(out["fp"], 3)
        self.assertAlmostEqual(out["specificity"], 37 / 40, places=4)

    def test_n77_bootstrap_median_is_0_38(self) -> None:
        # n=77 bootstrap (B=2000, seed=42) median should reproduce RQ44's 0.38.
        # The n=77 resample uses seed=42 (directly comparable to RQ44).
        idx = rq45.bootstrap_resample(77, 77, 200, 42)
        thrs = np.array([
            rq45.calibrate_threshold_at_spec(
                self.lang_ent[idx[b]], self.labels[idx[b]]
            )["threshold"]
            for b in range(200)
        ])
        self.assertAlmostEqual(float(np.median(thrs)), 0.38, places=2)

    def test_n77_interval_width_matches_rq44(self) -> None:
        # RQ44 reported interval width 0.94 ([0.01, 0.95]) at B=10000. With
        # B=2000 and seed=42 the width should also be ~0.94 (the extremes are
        # reached within the first 2000 resamples).
        idx = rq45.bootstrap_resample(77, 77, 2000, 42)
        thrs = np.array([
            rq45.calibrate_threshold_at_spec(
                self.lang_ent[idx[b]], self.labels[idx[b]]
            )["threshold"]
            for b in range(2000)
        ])
        w = rq45.percentile_interval_width(thrs, 2.5, 97.5)
        self.assertAlmostEqual(w, 0.94, delta=0.01)


# ---------------------------------------------------- convergence structural tests
class TestConvergenceStructure(unittest.TestCase):
    """Structural tests on the convergence direction (do not depend on the full
    B=2000 run, but use small-B samples to verify the monotonic trend)."""

    @classmethod
    def setUpClass(cls) -> None:
        data = json.loads(AISHELL4_JSON.read_text(encoding="utf-8"))
        windows = data["windows"]
        cls.lang_ent = np.array(
            [rq45.max_across_speakers(w) for w in windows], dtype=float
        )
        cls.sep_cpwer = np.array(
            [float(w["always_separated_cpwer"]) for w in windows], dtype=float
        )
        cls.mixed_cpwer = np.array(
            [float(w["always_mixed_cpwer"]) for w in windows], dtype=float
        )
        cls.labels = (cls.sep_cpwer > rq45.CATASTROPHIC_CPWER).astype(int)

    def _bootstrap_thresholds(self, n_sample: int, n_boot: int, seed: int) -> np.ndarray:
        idx = rq45.bootstrap_resample(77, n_sample, n_boot, seed)
        return np.array([
            rq45.calibrate_threshold_at_spec(
                self.lang_ent[idx[b]], self.labels[idx[b]]
            )["threshold"]
            for b in range(n_boot)
        ])

    def test_dominant_fraction_increases_with_n(self) -> None:
        # The 0.38 mode should become MORE dominant as n grows (the bad modes
        # require rare resample compositions that shrink with n).
        thrs_77 = self._bootstrap_thresholds(77, 500, 42)
        thrs_616 = self._bootstrap_thresholds(616, 500, 42 + 616)
        frac_77 = float(np.mean(thrs_77 == 0.38))
        frac_616 = float(np.mean(thrs_616 == 0.38))
        self.assertGreater(frac_616, frac_77)

    def test_n_modes_decreases_or_equal_with_n(self) -> None:
        # Modality (modes with >=5%) should not INCREASE as n grows.
        thrs_77 = self._bootstrap_thresholds(77, 1000, 42)
        thrs_616 = self._bootstrap_thresholds(616, 1000, 42 + 616)
        n_modes_77 = rq45.count_modes(thrs_77)["n_modes"]
        n_modes_616 = rq45.count_modes(thrs_616)["n_modes"]
        self.assertLessEqual(n_modes_616, n_modes_77)


if __name__ == "__main__":
    unittest.main()
