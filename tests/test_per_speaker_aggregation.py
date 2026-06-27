"""Tests for RQ56: per-speaker lang-id entropy aggregation comparison
(experimental/frontier).

Pins the pure helpers: ``per_speaker_entropies``, ``aggregate_scores``,
``aggregate_window``, ``build_adaptive_grid``, ``grid_for``,
``calibrate_threshold_at_spec``, ``corrected_cpwer``, ``bootstrap_indices``,
``out_of_bag_cpwer``, ``percentile_interval``, ``threshold_distribution``,
and the RQ13/RQ16/RQ25/RQ44-verbatim detector primitives
(``script_category``, ``language_id_entropy``). Also includes a smoke test
that the MAX arm reproduces RQ44's in-sample threshold (0.38, 35/37
sensitivity, 37/40 specificity) and cpWER (1.043) on the real 77-window
AISHELL-4 data, and sanity checks on the SUM / MEAN / MIN arms.

No Whisper / no ASR / no LLM / no audio needed. numpy + stdlib only.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# The RQ56 analysis script lives in results/frontier/ as a standalone module
# (no src. package). Import it via sys.path manipulation, mirroring the
# harness test_bootstrap_threshold / test_metadata_mode_s_detector pattern.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _PROJECT_ROOT / "results" / "frontier" / "per_speaker_aggregation_comparison"
sys.path.insert(0, str(_SCRIPT_DIR))

import per_speaker_aggregation_analysis as rq56  # noqa: E402  (path-injected import)

AISHELL4_JSON = (
    _PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)


# ------------------------------------------------------------- script_category
class TestScriptCategory(unittest.TestCase):
    def test_han(self) -> None:
        self.assertEqual(rq56.script_category("你"), "Han")

    def test_latin(self) -> None:
        self.assertEqual(rq56.script_category("A"), "Latin")

    def test_space(self) -> None:
        self.assertEqual(rq56.script_category(" "), "Space")

    def test_digit(self) -> None:
        self.assertEqual(rq56.script_category("7"), "Digit")

    def test_unknown_control_is_other(self) -> None:
        self.assertEqual(rq56.script_category("\x00"), "Other")


# ---------------------------------------------------------- language_id_entropy
class TestLanguageIdEntropy(unittest.TestCase):
    def test_monoscript_chinese_zero_entropy(self) -> None:
        self.assertEqual(rq56.language_id_entropy("你好世界"), 0.0)

    def test_mixed_script_positive_entropy(self) -> None:
        self.assertGreater(rq56.language_id_entropy("你好AB"), 0.5)

    def test_empty_is_zero(self) -> None:
        self.assertEqual(rq56.language_id_entropy(""), 0.0)

    def test_whitespace_only_is_zero(self) -> None:
        self.assertEqual(rq56.language_id_entropy("   \n\t  "), 0.0)

    def test_pure_latin_below_mixed(self) -> None:
        # Pure Latin is near-monoscript (entropy ~ 0), strictly less than a
        # 4-script mix at equal length.
        self.assertLess(
            rq56.language_id_entropy("hello world"),
            rq56.language_id_entropy("你好ABאあ"),
        )


# ---------------------------------------------------------- per_speaker_entropies
class TestPerSpeakerEntropies(unittest.TestCase):
    def test_filters_empty_and_whitespace_tracks(self) -> None:
        w = {"separated_text_per_speaker": {
            "001-M": "你好世界",     # monoscript -> 0.0 but non-empty -> kept
            "002-M": "   \n  ",       # whitespace -> skipped
            "003-F": "",              # empty -> skipped
            "004-F": None,            # None -> skipped
        }}
        ents = rq56.per_speaker_entropies(w)
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0], 0.0)

    def test_all_empty_returns_empty_list(self) -> None:
        w = {"separated_text_per_speaker": {"001-M": "", "002-M": "  "}}
        self.assertEqual(rq56.per_speaker_entropies(w), [])

    def test_missing_key_returns_empty(self) -> None:
        self.assertEqual(rq56.per_speaker_entropies({}), [])

    def test_multiple_speakers_preserved_in_dict_order(self) -> None:
        w = {"separated_text_per_speaker": {
            "001-M": "你好世界",   # 0.0
            "002-M": "你好AB",     # > 0.5
        }}
        ents = rq56.per_speaker_entropies(w)
        self.assertEqual(len(ents), 2)
        self.assertEqual(ents[0], 0.0)
        self.assertGreater(ents[1], 0.5)

    def test_mixed_script_value_is_float(self) -> None:
        w = {"separated_text_per_speaker": {"001-M": "你好ABא"}}
        ents = rq56.per_speaker_entropies(w)
        self.assertEqual(len(ents), 1)
        self.assertIsInstance(ents[0], float)


# --------------------------------------------------------------- aggregate_scores
class TestAggregateScores(unittest.TestCase):
    def test_max(self) -> None:
        self.assertEqual(rq56.aggregate_scores([0.1, 0.5, 0.2], "max"), 0.5)

    def test_sum(self) -> None:
        self.assertAlmostEqual(rq56.aggregate_scores([0.1, 0.5, 0.2], "sum"), 0.8)

    def test_mean(self) -> None:
        self.assertAlmostEqual(rq56.aggregate_scores([0.1, 0.5, 0.2], "mean"),
                               (0.1 + 0.5 + 0.2) / 3)

    def test_min(self) -> None:
        self.assertEqual(rq56.aggregate_scores([0.1, 0.5, 0.2], "min"), 0.1)

    def test_empty_list_all_return_zero(self) -> None:
        for agg in rq56.AGGREGATIONS:
            self.assertEqual(rq56.aggregate_scores([], agg), 0.0)

    def test_single_element_all_equal(self) -> None:
        for agg in rq56.AGGREGATIONS:
            self.assertAlmostEqual(rq56.aggregate_scores([0.42], agg), 0.42)

    def test_max_ge_mean_ge_min(self) -> None:
        vals = [0.1, 0.5, 0.2, 0.9, 0.0]
        mx = rq56.aggregate_scores(vals, "max")
        mn = rq56.aggregate_scores(vals, "min")
        mean = rq56.aggregate_scores(vals, "mean")
        self.assertGreaterEqual(mx, mean)
        self.assertGreaterEqual(mean, mn)

    def test_sum_scales_with_speaker_count(self) -> None:
        # SUM of n identical values = n * value; MAX/MEAN/MIN do not scale.
        v = 0.3
        self.assertAlmostEqual(rq56.aggregate_scores([v, v, v, v], "sum"), 1.2)
        self.assertAlmostEqual(rq56.aggregate_scores([v, v, v, v], "max"), 0.3)
        self.assertAlmostEqual(rq56.aggregate_scores([v, v, v, v], "mean"), 0.3)
        self.assertAlmostEqual(rq56.aggregate_scores([v, v, v, v], "min"), 0.3)

    def test_unknown_aggregation_raises(self) -> None:
        with self.assertRaises(ValueError):
            rq56.aggregate_scores([0.1, 0.2], "median")


# --------------------------------------------------------------- aggregate_window
class TestAggregateWindow(unittest.TestCase):
    def test_max_matches_rq44_convention(self) -> None:
        # MAX aggregation over per-speaker entropies == RQ44's max_across_speakers.
        w = {"separated_text_per_speaker": {
            "001-M": "你好世界",   # 0.0
            "002-M": "你好AB",     # high
        }}
        # RQ44 max_across_speakers semantics: max of the per-speaker entropies.
        expected = max(rq56.language_id_entropy("你好世界"),
                       rq56.language_id_entropy("你好AB"))
        self.assertAlmostEqual(rq56.aggregate_window(w, "max"), expected)

    def test_min_is_best_case_speaker(self) -> None:
        w = {"separated_text_per_speaker": {
            "001-M": "你好世界",   # 0.0
            "002-M": "你好AB",     # high
        }}
        self.assertEqual(rq56.aggregate_window(w, "min"), 0.0)

    def test_all_empty_returns_zero_for_every_aggregation(self) -> None:
        w = {"separated_text_per_speaker": {"001-M": "", "002-M": "  "}}
        for agg in rq56.AGGREGATIONS:
            self.assertEqual(rq56.aggregate_window(w, agg), 0.0)


# ------------------------------------------------------------ build_adaptive_grid
class TestBuildAdaptiveGrid(unittest.TestCase):
    def test_step_is_0_01(self) -> None:
        grid = rq56.build_adaptive_grid(np.array([0.0, 1.5]))
        diffs = [round(grid[i + 1] - grid[i], 6) for i in range(len(grid) - 1)]
        self.assertTrue(all(abs(d - 0.01) < 1e-9 for d in diffs))

    def test_starts_at_zero(self) -> None:
        self.assertEqual(rq56.build_adaptive_grid(np.array([2.5]))[0], 0.00)

    def test_covers_max_with_margin(self) -> None:
        # max=1.5, margin=1.05 -> 1.5*1.05=1.575 -> ceil=2 -> grid to 2.00
        grid = rq56.build_adaptive_grid(np.array([0.0, 1.5]))
        self.assertGreaterEqual(grid[-1], 1.5)
        self.assertEqual(grid[-1], 2.00)

    def test_all_zero_returns_single_zero(self) -> None:
        self.assertEqual(rq56.build_adaptive_grid(np.array([0.0, 0.0])), [0.00])

    def test_empty_returns_single_zero(self) -> None:
        self.assertEqual(rq56.build_adaptive_grid(np.array([])), [0.00])

    def test_minimum_top_is_one(self) -> None:
        # max=0.001 -> ceil(0.001*1.05)=1 -> grid 0.00..1.00 (101 points).
        # ``top`` is the integer ceiling in value units, so the grid runs to
        # 1.00 even for tiny observed maxima (harmless: the calibration just
        # picks a low threshold).
        grid = rq56.build_adaptive_grid(np.array([0.001]))
        self.assertEqual(grid[0], 0.00)
        self.assertEqual(grid[-1], 1.00)
        self.assertEqual(len(grid), 101)


# -------------------------------------------------------------------- grid_for
class TestGridFor(unittest.TestCase):
    def test_max_uses_fixed_rq44_grid(self) -> None:
        grid = rq56.grid_for("max", np.array([0.0, 1.5]))
        self.assertEqual(grid[0], 0.00)
        self.assertEqual(grid[-1], 2.00)
        self.assertEqual(len(grid), 201)

    def test_max_grid_ignores_observed_range(self) -> None:
        # Even if observed max is large, MAX still uses the 0.00-2.00 grid.
        grid = rq56.grid_for("max", np.array([0.0, 10.0]))
        self.assertEqual(grid[-1], 2.00)

    def test_non_max_uses_adaptive_grid(self) -> None:
        scores = np.array([0.0, 5.0])  # ceil(5*1.05)=ceil(5.25)=6 -> to 6.00
        grid = rq56.grid_for("sum", scores)
        self.assertEqual(grid[-1], 6.00)
        self.assertGreater(len(grid), 201)

    def test_mean_grid_adaptive(self) -> None:
        scores = np.array([0.0, 1.3])  # ceil(1.3*1.05)=ceil(1.365)=2 -> to 2.00
        grid = rq56.grid_for("mean", scores)
        self.assertEqual(grid[-1], 2.00)


# ------------------------------------------------------ calibrate_threshold_at_spec
class TestCalibrateThresholdAtSpec(unittest.TestCase):
    def test_separable_case(self) -> None:
        scores = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 1.0, 1.1])
        labels = np.array([0, 0, 0, 0, 0, 1, 1])
        out = rq56.calibrate_threshold_at_spec(scores, labels,
                                               grid=[round(0.01 * i, 2) for i in range(0, 201)])
        self.assertGreaterEqual(out["specificity"], 0.90 - 1e-9)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["tp"], 2)
        self.assertEqual(out["fn"], 0)

    def test_target_specificity_respected(self) -> None:
        scores = np.array([0., 0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1])
        out = rq56.calibrate_threshold_at_spec(scores, labels, target_spec=0.75)
        self.assertGreaterEqual(out["specificity"], 0.75 - 1e-9)

    def test_no_threshold_meets_target_falls_back(self) -> None:
        scores = np.array([0., 1., 2., 3.])
        labels = np.array([0, 1, 0, 1])
        out = rq56.calibrate_threshold_at_spec(scores, labels, target_spec=0.99)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertEqual(out["tp"], 0)
        self.assertEqual(out["fn"], 2)
        self.assertEqual(out["specificity"], 1.0)

    def test_empty_positives_safe(self) -> None:
        scores = np.array([0., 1., 2.])
        labels = np.array([0, 0, 0])
        out = rq56.calibrate_threshold_at_spec(scores, labels)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertGreaterEqual(out["specificity"], 0.90 - 1e-9)

    def test_tie_break_lower_threshold(self) -> None:
        scores = np.array([0., 1., 5., 6.])
        labels = np.array([0, 0, 1, 1])
        out = rq56.calibrate_threshold_at_spec(scores, labels, target_spec=0.5)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["specificity"], 1.0)
        self.assertAlmostEqual(out["threshold"], 1.01, places=6)

    def test_returns_all_confusion_counts(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq56.calibrate_threshold_at_spec(scores, labels, target_spec=0.5)
        for key in ("threshold", "sensitivity", "specificity",
                    "tp", "fp", "tn", "fn"):
            self.assertIn(key, out)
        self.assertEqual(out["tp"] + out["fn"], 2)
        self.assertEqual(out["fp"] + out["tn"], 2)

    def test_custom_grid_used(self) -> None:
        # A coarse custom grid; threshold must be one of its values.
        scores = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        grid = [0.0, 0.5, 1.0]
        out = rq56.calibrate_threshold_at_spec(scores, labels, grid=grid)
        self.assertIn(out["threshold"], grid)


# --------------------------------------------------------------- corrected_cpwer
class TestCorrectedCpwer(unittest.TestCase):
    def test_routes_mixed_when_above_threshold(self) -> None:
        scores = np.array([0.9, 0.1])
        mixed = np.array([1.0, 1.0])
        sep = np.array([2.0, 2.0])
        # window0: 0.9>=0.5 -> mixed 1.0 ; window1: 0.1<0.5 -> sep 2.0 ; mean=1.5
        self.assertAlmostEqual(rq56.corrected_cpwer(scores, mixed, sep, 0.5), 1.5)

    def test_all_below_threshold_routes_all_separated(self) -> None:
        scores = np.array([0.1, 0.2])
        mixed = np.array([1.0, 1.0])
        sep = np.array([2.0, 3.0])
        self.assertAlmostEqual(rq56.corrected_cpwer(scores, mixed, sep, 0.5), 2.5)

    def test_all_above_threshold_routes_all_mixed(self) -> None:
        scores = np.array([0.9, 0.8])
        mixed = np.array([1.0, 2.0])
        sep = np.array([3.0, 3.0])
        self.assertAlmostEqual(rq56.corrected_cpwer(scores, mixed, sep, 0.5), 1.5)

    def test_threshold_boundary_inclusive(self) -> None:
        # score == threshold is flagged (>= with EPS).
        scores = np.array([0.5, 0.5])
        mixed = np.array([1.0, 1.0])
        sep = np.array([3.0, 3.0])
        self.assertAlmostEqual(rq56.corrected_cpwer(scores, mixed, sep, 0.5), 1.0)

    def test_empty_returns_nan(self) -> None:
        self.assertTrue(math.isnan(
            rq56.corrected_cpwer(np.array([]), np.array([]), np.array([]), 0.5)))


# ------------------------------------------------------------- bootstrap_indices
class TestBootstrapIndices(unittest.TestCase):
    def test_shape(self) -> None:
        idx = rq56.bootstrap_indices(77, 100, 42)
        self.assertEqual(idx.shape, (100, 77))

    def test_values_in_range(self) -> None:
        idx = rq56.bootstrap_indices(77, 100, 42)
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 77))

    def test_deterministic_given_seed(self) -> None:
        a = rq56.bootstrap_indices(10, 5, 42)
        b = rq56.bootstrap_indices(10, 5, 42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = rq56.bootstrap_indices(10, 5, 42)
        b = rq56.bootstrap_indices(10, 5, 7)
        self.assertFalse(np.array_equal(a, b))


# --------------------------------------------------------------- out_of_bag_cpwer
class TestOutOfBagCpwer(unittest.TestCase):
    def test_oob_excludes_in_bag(self) -> None:
        # n=5, in_bag = [0,0,1,1,2] -> OOB = {3,4}
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        mixed = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sep = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        out = rq56.out_of_bag_cpwer(scores, mixed, sep, threshold=0.4,
                                    in_bag_idx=np.array([0, 0, 1, 1, 2]))
        self.assertEqual(out["n_oob"], 2)
        self.assertEqual(out["n_flagged_mixed"], 2)
        self.assertEqual(out["n_separated"], 0)
        self.assertAlmostEqual(out["cpwer"], 1.0)

    def test_routes_separated_when_below_threshold(self) -> None:
        scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        mixed = np.array([1.0] * 5)
        sep = np.array([2.0] * 5)
        out = rq56.out_of_bag_cpwer(scores, mixed, sep, threshold=0.4,
                                    in_bag_idx=np.array([0, 1, 2]))  # OOB={3,4}
        self.assertEqual(out["n_oob"], 2)
        self.assertEqual(out["n_separated"], 2)
        self.assertAlmostEqual(out["cpwer"], 2.0)

    def test_all_in_bag_returns_nan(self) -> None:
        scores = np.array([0.5, 0.5])
        mixed = np.array([1.0, 1.0])
        sep = np.array([2.0, 2.0])
        out = rq56.out_of_bag_cpwer(scores, mixed, sep, threshold=0.4,
                                    in_bag_idx=np.array([0, 1, 0, 1]))
        self.assertEqual(out["n_oob"], 0)
        self.assertTrue(math.isnan(out["cpwer"]))


# ----------------------------------------------------------- percentile_interval
class TestPercentileInterval(unittest.TestCase):
    def test_matches_numpy_percentile(self) -> None:
        vals = np.arange(101.0)
        lo, hi = rq56.percentile_interval(vals, 2.5, 97.5)
        self.assertAlmostEqual(lo, np.percentile(vals, 2.5))
        self.assertAlmostEqual(hi, np.percentile(vals, 97.5))

    def test_empty_returns_nan(self) -> None:
        lo, hi = rq56.percentile_interval([])
        self.assertTrue(math.isnan(lo))
        self.assertTrue(math.isnan(hi))


# ----------------------------------------------------------- threshold_distribution
class TestThresholdDistribution(unittest.TestCase):
    def test_n_unique_counts_distinct_values(self) -> None:
        thr = np.array([0.38, 0.38, 0.38, 0.01, 0.95])
        out = rq56.threshold_distribution(thr)
        self.assertEqual(out["n_unique"], 3)

    def test_mode_is_most_frequent(self) -> None:
        thr = np.array([0.38, 0.38, 0.38, 0.01, 0.95])
        out = rq56.threshold_distribution(thr)
        self.assertEqual(out["mode"], 0.38)
        self.assertEqual(out["mode_count"], 3)
        self.assertAlmostEqual(out["mode_fraction"], 3 / 5)

    def test_empty_distribution(self) -> None:
        out = rq56.threshold_distribution(np.array([]))
        self.assertEqual(out["n_unique"], 0)
        self.assertEqual(out["mode_count"], 0)

    def test_modes_within_10pct_captures_ties(self) -> None:
        # Two values with equal counts -> both within 10% of max.
        thr = np.array([0.38, 0.38, 0.01, 0.01])
        out = rq56.threshold_distribution(thr)
        self.assertEqual(len(out["modes_within_10pct"]), 2)

    def test_top_values_length(self) -> None:
        thr = np.array([0.38, 0.01, 0.95, 0.84, 0.87, 0.33])
        out = rq56.threshold_distribution(thr, top_k=3)
        self.assertEqual(len(out["top_values"]), 3)


# ------------------------------------------------------------------- count_modes
class TestCountModes(unittest.TestCase):
    def test_counts_modes_above_min_fraction(self) -> None:
        # 0.38 appears 60% (>=5%), 0.87 appears 15% (>=5%), 0.01 9% (>=5%),
        # 0.95 8% (>=5%), 0.33 5.7% (>=5%), 0.84 2.3% (<5% -> not a mode).
        thr = np.array([0.38] * 60 + [0.87] * 15 + [0.01] * 9
                       + [0.95] * 8 + [0.33] * 6 + [0.84] * 2)
        out = rq56.count_modes(thr, 0.05)
        self.assertEqual(out["n_modes"], 5)
        self.assertNotIn(0.84, [m["threshold"] for m in out["modes"]])

    def test_modes_sorted_by_descending_count(self) -> None:
        thr = np.array([0.38] * 60 + [0.87] * 15 + [0.01] * 9)
        out = rq56.count_modes(thr, 0.05)
        counts = [m["count"] for m in out["modes"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(out["modes"][0]["threshold"], 0.38)

    def test_tie_break_ascending_threshold(self) -> None:
        # Equal counts -> stable sort keeps ascending-threshold order.
        thr = np.array([0.38, 0.38, 0.01, 0.01])
        out = rq56.count_modes(thr, 0.05)
        self.assertEqual(out["n_modes"], 2)
        self.assertEqual(out["modes"][0]["threshold"], 0.01)
        self.assertEqual(out["modes"][1]["threshold"], 0.38)

    def test_n_unique_reports_all_distinct(self) -> None:
        thr = np.array([0.38] * 60 + [0.87] * 15 + [0.84] * 2)
        out = rq56.count_modes(thr, 0.05)
        # n_unique counts ALL distinct values (3), not just >= 5% modes (2).
        self.assertEqual(out["n_unique"], 3)
        self.assertEqual(out["n_modes"], 2)

    def test_empty_returns_zero(self) -> None:
        out = rq56.count_modes(np.array([]), 0.05)
        self.assertEqual(out["n_modes"], 0)
        self.assertEqual(out["n_unique"], 0)
        self.assertEqual(out["modes"], [])

    def test_all_below_threshold_returns_zero_modes(self) -> None:
        # Every value appears once (1% each, n=100) -> none reach 5%.
        thr = np.arange(100, dtype=float)
        out = rq56.count_modes(thr, 0.05)
        self.assertEqual(out["n_modes"], 0)
        self.assertEqual(out["n_unique"], 100)

    def test_min_fraction_included_in_output(self) -> None:
        out = rq56.count_modes(np.array([0.38, 0.38, 0.01]), 0.05)
        self.assertEqual(out["min_fraction"], 0.05)

    def test_custom_min_fraction(self) -> None:
        # With min_fraction=0.5, only the 60% value qualifies.
        thr = np.array([0.38] * 60 + [0.87] * 40)
        out = rq56.count_modes(thr, 0.50)
        self.assertEqual(out["n_modes"], 1)
        self.assertEqual(out["modes"][0]["threshold"], 0.38)

    def test_boundary_5pct_inclusive(self) -> None:
        # Exactly 5% (5 of 100) is included (>= with EPS tolerance).
        thr = np.array([0.38] * 95 + [0.01] * 5)
        out = rq56.count_modes(thr, 0.05)
        self.assertEqual(out["n_modes"], 2)

    def test_fraction_field_correct(self) -> None:
        thr = np.array([0.38] * 60 + [0.87] * 40)
        out = rq56.count_modes(thr, 0.05)
        self.assertAlmostEqual(out["modes"][0]["fraction"], 0.60)
        self.assertAlmostEqual(out["modes"][1]["fraction"], 0.40)

    def test_matches_rq48_count_modes(self) -> None:
        # Cross-check against RQ48's count_modes (the source of the verbatim copy).
        rq48_dir = (_PROJECT_ROOT / "results" / "frontier"
                    / "calibration_rule_comparison")
        sys.path.insert(0, str(rq48_dir))
        try:
            import calibration_rule_analysis as rq48  # noqa: E402
        except ImportError:
            self.skipTest("RQ48 calibration_rule_analysis not importable")
        rng = np.random.default_rng(123)
        thr = rng.choice([0.01, 0.33, 0.38, 0.84, 0.87, 0.95],
                         size=10000, p=[0.09, 0.06, 0.60, 0.02, 0.15, 0.08])
        a = rq56.count_modes(thr, 0.05)
        b = rq48.count_modes(thr, 0.05)
        self.assertEqual(a["n_modes"], b["n_modes"])
        self.assertEqual([m["threshold"] for m in a["modes"]],
                         [m["threshold"] for m in b["modes"]])


# ------------------------------------------------------------- module constants
class TestModuleConstants(unittest.TestCase):
    def test_aggregations_set(self) -> None:
        self.assertEqual(set(rq56.AGGREGATIONS), {"max", "sum", "mean", "min"})

    def test_aggregations_order(self) -> None:
        self.assertEqual(rq56.AGGREGATIONS, ("max", "sum", "mean", "min"))

    def test_bootstrap_config(self) -> None:
        self.assertEqual(rq56.N_BOOT, 10000)
        self.assertEqual(rq56.SEED, 42)

    def test_target_specificity(self) -> None:
        self.assertEqual(rq56.TARGET_SPECIFICITY, 0.90)

    def test_max_grid_is_rq44_exact(self) -> None:
        self.assertEqual(rq56.MAX_GRID[0], 0.00)
        self.assertEqual(rq56.MAX_GRID[-1], 2.00)
        self.assertEqual(len(rq56.MAX_GRID), 201)

    def test_h56c_kill_threshold(self) -> None:
        self.assertEqual(rq56.H56C_MAX_CPWER, 1.10)

    def test_min_mode_fraction(self) -> None:
        self.assertEqual(rq56.MIN_MODE_FRACTION, 0.05)

    def test_catastrophic_cpwer(self) -> None:
        self.assertEqual(rq56.CATASTROPHIC_CPWER, 1.0)


# ----------------------------------------------- in-sample reproduction (RQ44 MAX)
@unittest.skipUnless(AISHELL4_JSON.exists(), "AISHELL-4 validation JSON not present")
class TestInSampleReproduction(unittest.TestCase):
    """Validates that the MAX arm reproduces RQ44's in-sample threshold (0.38,
    35/37 sensitivity, 37/40 specificity, cpWER 1.043) on the real 77-window
    AISHELL-4 data, and sanity-checks the SUM / MEAN / MIN arms."""

    @classmethod
    def setUpClass(cls) -> None:
        data = json.loads(AISHELL4_JSON.read_text(encoding="utf-8"))
        windows = data["windows"]
        cls.windows = windows
        cls.mixed = np.array([float(w["always_mixed_cpwer"]) for w in windows], dtype=float)
        cls.sep = np.array([float(w["always_separated_cpwer"]) for w in windows], dtype=float)
        cls.labels = (cls.sep > 1.0).astype(int)
        cls.n = len(windows)
        cls.per_speaker = [rq56.per_speaker_entropies(w) for w in windows]

    def test_77_windows(self) -> None:
        self.assertEqual(self.n, 77)

    def test_37_hallucinated_40_clean(self) -> None:
        self.assertEqual(int(self.labels.sum()), 37)
        self.assertEqual(int((self.labels == 0).sum()), 40)

    def test_max_in_sample_threshold_is_038(self) -> None:
        scores = np.array([rq56.aggregate_scores(p, "max") for p in self.per_speaker],
                          dtype=float)
        out = rq56.calibrate_threshold_at_spec(scores, self.labels,
                                               grid=list(rq56.MAX_GRID))
        self.assertAlmostEqual(out["threshold"], 0.38, places=6)

    def test_max_in_sample_operating_point_matches_rq44(self) -> None:
        scores = np.array([rq56.aggregate_scores(p, "max") for p in self.per_speaker],
                          dtype=float)
        out = rq56.calibrate_threshold_at_spec(scores, self.labels,
                                               grid=list(rq56.MAX_GRID))
        self.assertAlmostEqual(out["sensitivity"], 35 / 37, places=4)
        self.assertAlmostEqual(out["specificity"], 37 / 40, places=4)
        self.assertEqual(out["tp"], 35)
        self.assertEqual(out["fp"], 3)

    def test_max_in_sample_cpwer_matches_rq44(self) -> None:
        scores = np.array([rq56.aggregate_scores(p, "max") for p in self.per_speaker],
                          dtype=float)
        out = rq56.calibrate_threshold_at_spec(scores, self.labels,
                                               grid=list(rq56.MAX_GRID))
        cpwer = rq56.corrected_cpwer(scores, self.mixed, self.sep, out["threshold"])
        self.assertAlmostEqual(cpwer, 1.043, places=2)

    def test_all_four_aggregations_calibrate(self) -> None:
        # Every aggregation must produce a valid operating point at >= 90% spec.
        for agg in rq56.AGGREGATIONS:
            scores = np.array([rq56.aggregate_scores(p, agg) for p in self.per_speaker],
                              dtype=float)
            grid = rq56.grid_for(agg, scores)
            out = rq56.calibrate_threshold_at_spec(scores, self.labels, grid=grid)
            self.assertGreaterEqual(out["specificity"], 0.90 - 1e-9,
                                    msg=f"{agg} specificity below target")
            self.assertIn("threshold", out)

    def test_sum_scores_ge_max_scores(self) -> None:
        # SUM >= MAX elementwise (sum of nonneg values >= their max).
        for p in self.per_speaker:
            if not p:
                continue
            self.assertGreaterEqual(rq56.aggregate_scores(p, "sum"),
                                    rq56.aggregate_scores(p, "max"))

    def test_max_scores_ge_mean_scores_ge_min_scores(self) -> None:
        for p in self.per_speaker:
            if not p:
                continue
            mx = rq56.aggregate_scores(p, "max")
            mean = rq56.aggregate_scores(p, "mean")
            mn = rq56.aggregate_scores(p, "min")
            self.assertGreaterEqual(mx, mean)
            self.assertGreaterEqual(mean, mn)

    def test_run_arm_includes_count_modes(self) -> None:
        # Small bootstrap (B=500) on the MAX arm: verify run_aggregation_arm
        # emits n_modes_5pct + modes_5pct and that n_modes_5pct <= n_unique.
        scores = np.array([rq56.aggregate_scores(p, "max") for p in self.per_speaker],
                          dtype=float)
        boot_idx = rq56.bootstrap_indices(self.n, 500, 42)
        arm = rq56.run_aggregation_arm(
            "max", scores, self.labels, self.mixed, self.sep,
            list(rq56.MAX_GRID), boot_idx,
        )
        td = arm["threshold_distribution"]
        self.assertIn("n_modes_5pct", td)
        self.assertIn("modes_5pct", td)
        self.assertLessEqual(td["n_modes_5pct"], td["n_unique"])
        # Every reported mode must have fraction >= MIN_MODE_FRACTION - EPS.
        for m in td["modes_5pct"]:
            self.assertGreaterEqual(m["fraction"], rq56.MIN_MODE_FRACTION - 1e-9)

    def test_h56b_uses_count_modes_metric(self) -> None:
        # MAX and SUM share the same 6 distinct in-sample threshold candidates
        # on this data; with B=2000 their n_modes_5pct must be computable and
        # the H56b kill metric (n_modes_5pct) must be <= n_unique for both.
        boot_idx = rq56.bootstrap_indices(self.n, 2000, 42)
        for agg in ("max", "sum"):
            scores = np.array([rq56.aggregate_scores(p, agg) for p in self.per_speaker],
                              dtype=float)
            grid = rq56.grid_for(agg, scores)
            arm = rq56.run_aggregation_arm(
                agg, scores, self.labels, self.mixed, self.sep, grid, boot_idx,
            )
            td = arm["threshold_distribution"]
            self.assertLessEqual(td["n_modes_5pct"], td["n_unique"])


if __name__ == "__main__":
    unittest.main()
