"""Tests for RQ38 speaker-count-effect analysis (experimental/frontier, issue #945).

Pin the PURE helpers: active-speaker counting, silence fraction, length ratio,
hallucination label, Mode S membership, feature extraction, Spearman correlation,
rank-based partial correlation, permutation test, bootstrap rate CI, and
stratification. No IO is exercised — the tests run against synthetic data and the
helpers in
``results/frontier/speaker_count_effect/speaker_count_effect_analysis.py``.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

# Load the analysis module by absolute path (it lives under results/, not a package).
_MOD_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "frontier"
    / "speaker_count_effect"
    / "speaker_count_effect_analysis.py"
)
_spec = importlib.util.spec_from_file_location("speaker_count_effect_analysis", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["speaker_count_effect_analysis"] = _mod

count_active_speakers = _mod.count_active_speakers
compute_silence_fraction = _mod.compute_silence_fraction
compute_length_ratio = _mod.compute_length_ratio
is_hallucinated = _mod.is_hallucinated
is_mode_s_window = _mod.is_mode_s_window
extract_window_features = _mod.extract_window_features
spearman_rho = _mod.spearman_rho
partial_correlation = _mod.partial_correlation
permutation_test_spearman = _mod.permutation_test_spearman
bootstrap_rate_ci = _mod.bootstrap_rate_ci
stratify_by_count = _mod.stratify_by_count
evaluate_hypotheses = _mod.evaluate_hypotheses


# ===========================================================================
# Active-speaker counting
# ===========================================================================
class TestCountActiveSpeakers(unittest.TestCase):
    def test_all_non_empty(self) -> None:
        self.assertEqual(count_active_speakers({"a": "hi", "b": "yo"}), 2)

    def test_all_empty_strings(self) -> None:
        self.assertEqual(count_active_speakers({"a": "", "b": ""}), 0)

    def test_mixed_empty_and_non_empty(self) -> None:
        self.assertEqual(count_active_speakers({"a": "hi", "b": "", "c": "yo"}), 2)

    def test_whitespace_only_counts_as_empty(self) -> None:
        self.assertEqual(count_active_speakers({"a": "   \t\n", "b": "hi"}), 1)

    def test_empty_dict(self) -> None:
        self.assertEqual(count_active_speakers({}), 0)

    def test_none_value_treated_as_empty(self) -> None:
        # Real JSON never yields None here, but the helper must not crash on it.
        self.assertEqual(count_active_speakers({"a": None, "b": "x"}), 1)


# ===========================================================================
# Silence fraction
# ===========================================================================
class TestSilenceFraction(unittest.TestCase):
    def test_all_active_returns_zero(self) -> None:
        self.assertAlmostEqual(compute_silence_fraction(3, 3), 0.0)

    def test_none_active_returns_one(self) -> None:
        self.assertAlmostEqual(compute_silence_fraction(4, 0), 1.0)

    def test_partial(self) -> None:
        # 4 present, 1 active -> 3/4 silent
        self.assertAlmostEqual(compute_silence_fraction(4, 1), 0.75)

    def test_zero_speakers_returns_zero(self) -> None:
        # No speakers present -> no speaker-attributable silence.
        self.assertAlmostEqual(compute_silence_fraction(0, 0), 0.0)

    def test_active_clamped_to_num_speakers(self) -> None:
        # Negative or over-count active is clamped, never producing fractions <0 or >1.
        self.assertAlmostEqual(compute_silence_fraction(3, 5), 0.0)
        self.assertAlmostEqual(compute_silence_fraction(3, -1), 1.0)


# ===========================================================================
# Length ratio
# ===========================================================================
class TestLengthRatio(unittest.TestCase):
    def test_basic_ratio(self) -> None:
        self.assertAlmostEqual(compute_length_ratio(100, 50), 2.0)

    def test_zero_mixed_length_returns_zero(self) -> None:
        self.assertAlmostEqual(compute_length_ratio(50, 0), 0.0)

    def test_zero_separated_length_returns_zero(self) -> None:
        self.assertAlmostEqual(compute_length_ratio(0, 50), 0.0)


# ===========================================================================
# Hallucination label
# ===========================================================================
class TestIsHallucinated(unittest.TestCase):
    def test_above_threshold(self) -> None:
        self.assertTrue(is_hallucinated(1.5))

    def test_exactly_threshold_is_not_hallucinated(self) -> None:
        # Strict >: the all-empty-decode cpWER of exactly 1.0 is non-hallucinated.
        self.assertFalse(is_hallucinated(1.0))

    def test_below_threshold(self) -> None:
        self.assertFalse(is_hallucinated(0.8))

    def test_custom_threshold(self) -> None:
        self.assertTrue(is_hallucinated(1.5, threshold=1.4))
        self.assertFalse(is_hallucinated(1.4, threshold=1.4))


# ===========================================================================
# Mode S membership
# ===========================================================================
class TestIsModeS(unittest.TestCase):
    def test_mode_s_id_and_hallucinated(self) -> None:
        self.assertTrue(is_mode_s_window(22, True))

    def test_mode_s_id_but_not_hallucinated(self) -> None:
        # Mode S requires hallucination; a clean w22 is not Mode S.
        self.assertFalse(is_mode_s_window(22, False))

    def test_non_mode_s_id(self) -> None:
        self.assertFalse(is_mode_s_window(5, True))


# ===========================================================================
# Feature extraction
# ===========================================================================
class TestExtractFeatures(unittest.TestCase):
    def _window(self, **overrides) -> dict:
        base = {
            "window_id": 0,
            "num_speakers": 2,
            "separated_text_per_speaker": {"001-M": "hello", "002-M": "world"},
            "separated_total_length": 10,
            "mixed_text_length": 10,
            "runtime_ratio": 1.0,
            "always_separated_cpwer": 0.5,
            "always_mixed_cpwer": 0.7,
            "overlap_label": "NoOverlap",
        }
        base.update(overrides)
        return base

    def test_full_window_all_active(self) -> None:
        f = extract_window_features(self._window())
        self.assertEqual(f["active_speakers"], 2)
        self.assertEqual(f["empty_speakers"], 0)
        self.assertAlmostEqual(f["silence_fraction"], 0.0)
        self.assertAlmostEqual(f["length_ratio"], 1.0)
        self.assertFalse(f["hallucinated"])
        self.assertFalse(f["mode_s"])

    def test_window_with_empty_speaker(self) -> None:
        f = extract_window_features(
            self._window(
                window_id=22,
                num_speakers=2,
                separated_text_per_speaker={"005-F": "abc", "006-F": ""},
                separated_total_length=3,
                always_separated_cpwer=2.0,
            )
        )
        self.assertEqual(f["active_speakers"], 1)
        self.assertEqual(f["empty_speakers"], 1)
        self.assertAlmostEqual(f["silence_fraction"], 0.5)
        self.assertTrue(f["hallucinated"])
        self.assertTrue(f["mode_s"])  # window 22 + hallucinated

    def test_all_empty_speakers_is_non_hallucinated(self) -> None:
        f = extract_window_features(
            self._window(
                num_speakers=1,
                separated_text_per_speaker={"001-M": ""},
                separated_total_length=0,
                always_separated_cpwer=1.0,
            )
        )
        self.assertEqual(f["active_speakers"], 0)
        self.assertAlmostEqual(f["silence_fraction"], 1.0)
        self.assertFalse(f["hallucinated"])  # structural floor


# ===========================================================================
# Spearman correlation
# ===========================================================================
class TestSpearman(unittest.TestCase):
    def test_perfect_monotonic(self) -> None:
        rho, p = spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        self.assertAlmostEqual(rho, 1.0, places=6)

    def test_perfect_inverse(self) -> None:
        rho, _ = spearman_rho([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertAlmostEqual(rho, -1.0, places=6)

    def test_constant_column_returns_zero(self) -> None:
        rho, p = spearman_rho([2, 2, 2, 2], [1, 2, 3, 4])
        self.assertEqual(rho, 0.0)
        self.assertEqual(p, 1.0)

    def test_short_input_returns_zero(self) -> None:
        rho, p = spearman_rho([1.0], [2.0])
        self.assertEqual(rho, 0.0)
        self.assertEqual(p, 1.0)


# ===========================================================================
# Partial correlation
# ===========================================================================
class TestPartialCorrelation(unittest.TestCase):
    def test_z_fully_mediates_gives_near_zero(self) -> None:
        # A proper x -> z -> y mediation chain: y depends on x ONLY through z.
        # Controlling for z should remove the x-y correlation. Continuous-random
        # values (not a sequence) so rank-based partial correlation has genuine
        # rank variation to work with.
        rng = np.random.default_rng(0)
        n = 400
        x = rng.uniform(0, 100, n)
        z = x + rng.normal(0, 3.0, n)   # z driven by x, noise causes rank jitter
        y = z + rng.normal(0, 3.0, n)   # y driven by z (mediated), not directly by x
        # Sanity: x and y are strongly correlated unconditionally.
        rho_full, _ = spearman_rho(x, y)
        self.assertGreater(rho_full, 0.9)
        partial = partial_correlation(x, y, z)
        # Mediation should remove most of the correlation.
        self.assertLess(abs(partial), 0.15)
        self.assertLess(abs(partial), abs(rho_full) * 0.2)

    def test_perfect_collinearity_returns_zero(self) -> None:
        # When z is identical to x, partial correlation is undefined (zero
        # residual variance). The helper must return 0.0, not numerical noise.
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        z = list(x)  # z == x exactly
        y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        partial = partial_correlation(x, y, z)
        self.assertEqual(partial, 0.0)

    def test_z_independent_keeps_correlation(self) -> None:
        # x and y strongly correlated; z independent of both -> partial ~ full.
        rng = np.random.default_rng(1)
        x = np.arange(60, dtype=float)
        y = x + rng.normal(0, 1.0, size=60)
        z = rng.normal(0, 1.0, size=60)
        rho_full, _ = spearman_rho(x, y)
        partial = partial_correlation(x, y, z)
        # Partial should be close to the full rank correlation (z carries no shared info).
        self.assertGreater(partial, rho_full - 0.2)

    def test_constant_z_falls_back_to_raw_rank_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        z = [7.0, 7.0, 7.0, 7.0, 7.0]
        partial = partial_correlation(x, y, z)
        rho_full, _ = spearman_rho(x, y)
        self.assertAlmostEqual(partial, rho_full, places=4)

    def test_short_input_returns_zero(self) -> None:
        self.assertEqual(partial_correlation([1.0, 2.0], [2.0, 3.0], [1.0, 1.0]), 0.0)


# ===========================================================================
# Permutation test
# ===========================================================================
class TestPermutationTest(unittest.TestCase):
    def test_strong_effect_small_p(self) -> None:
        x = list(range(40))
        y = list(range(40))  # perfect correlation
        rho, p = permutation_test_spearman(x, y, n_perm=500, seed=42)
        self.assertGreater(rho, 0.99)
        self.assertLess(p, 0.05)

    def test_no_effect_large_p(self) -> None:
        rng = np.random.default_rng(7)
        x = list(range(60))
        y = list(rng.permutation(60))  # y independent of x
        rho, p = permutation_test_spearman(x, y, n_perm=500, seed=42)
        self.assertGreater(p, 0.05)

    def test_seed_reproducibility(self) -> None:
        x = list(range(30))
        y = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 3
        r1, p1 = permutation_test_spearman(x, y, n_perm=300, seed=42)
        r2, p2 = permutation_test_spearman(x, y, n_perm=300, seed=42)
        self.assertEqual(r1, r2)
        self.assertEqual(p1, p2)

    def test_p_value_bounded_by_smoothing(self) -> None:
        # p-value uses +1 smoothing -> never exactly 0.
        x = list(range(30))
        y = list(range(30))
        _, p = permutation_test_spearman(x, y, n_perm=200, seed=42)
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0)


# ===========================================================================
# Bootstrap rate CI
# ===========================================================================
class TestBootstrapRateCI(unittest.TestCase):
    def test_all_true_rate_one(self) -> None:
        rate, lo, hi = bootstrap_rate_ci([1, 1, 1, 1], n_boot=500, seed=42)
        self.assertAlmostEqual(rate, 1.0)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_all_false_rate_zero(self) -> None:
        rate, lo, hi = bootstrap_rate_ci([0, 0, 0], n_boot=500, seed=42)
        self.assertAlmostEqual(rate, 0.0)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0)

    def test_mixed_rate_within_unit_interval(self) -> None:
        rate, lo, hi = bootstrap_rate_ci([1, 0, 1, 0, 1], n_boot=1000, seed=42)
        self.assertAlmostEqual(rate, 0.6, places=6)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, rate)
        self.assertGreaterEqual(hi, rate)

    def test_seed_reproducibility(self) -> None:
        labels = [1, 0, 1, 1, 0, 0, 1, 0]
        r1, lo1, hi1 = bootstrap_rate_ci(labels, n_boot=400, seed=42)
        r2, lo2, hi2 = bootstrap_rate_ci(labels, n_boot=400, seed=42)
        self.assertEqual((r1, lo1, hi1), (r2, lo2, hi2))

    def test_empty_input_returns_zeros(self) -> None:
        rate, lo, hi = bootstrap_rate_ci([])
        self.assertEqual((rate, lo, hi), (0.0, 0.0, 0.0))


# ===========================================================================
# Stratification
# ===========================================================================
class TestStratify(unittest.TestCase):
    def test_correct_grouping_and_rate(self) -> None:
        # active: 0,0,1,2,2,3  labels: 0,0,1,1,0,1
        active = [0, 0, 1, 2, 2, 3]
        labels = [0, 0, 1, 1, 0, 1]
        rows = stratify_by_count(
            active, labels, strata=((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")),
            n_boot=200, seed=42,
        )
        by = {r["stratum"]: r for r in rows}
        self.assertEqual(by["0"]["n"], 2)
        self.assertEqual(by["0"]["halluc"], 0)
        self.assertAlmostEqual(by["0"]["rate"], 0.0)
        self.assertEqual(by["1"]["n"], 1)
        self.assertAlmostEqual(by["1"]["rate"], 1.0)
        self.assertEqual(by["2"]["n"], 2)
        self.assertAlmostEqual(by["2"]["rate"], 0.5)
        self.assertEqual(by["3+"]["n"], 1)
        self.assertAlmostEqual(by["3+"]["rate"], 1.0)

    def test_empty_stratum_returns_nan_rate(self) -> None:
        rows = stratify_by_count([1, 2], [1, 0], strata=((0, 0, "0"), (1, 2, "1+")),
                                 n_boot=100, seed=42)
        by = {r["stratum"]: r for r in rows}
        self.assertEqual(by["0"]["n"], 0)
        self.assertTrue(np.isnan(by["0"]["rate"]))

    def test_3plus_bucket_collapses_high_counts(self) -> None:
        active = [3, 4, 6, 5]
        labels = [1, 1, 0, 1]
        rows = stratify_by_count(active, labels, n_boot=200, seed=42)
        top = [r for r in rows if r["stratum"] == "3+"][0]
        self.assertEqual(top["n"], 4)
        self.assertEqual(top["halluc"], 3)


# ===========================================================================
# Hypothesis evaluation (pure decision logic)
# ===========================================================================
class TestEvaluateHypotheses(unittest.TestCase):
    def test_h38a_supported_when_rho_above_threshold_and_p_small(self) -> None:
        v = evaluate_hypotheses(
            rho_active_cpwer=0.5, perm_p_active_cpwer=0.001,
            rho_active_halluc=0.4, perm_p_active_halluc=0.01,
            rho_num_halluc=0.3,
            mode_s_active_counts={22: 1, 30: 1},
            mode_s_num_counts={22: 2, 30: 1},
            partial_active_silence=0.05,
            partial_active_lengthratio=0.2,
            partial_active_silence_active_ge1=0.1,
        )
        self.assertEqual(v["H38a"]["verdict"], "SUPPORTED")
        self.assertTrue(v["H38a"]["supported"])

    def test_h38a_not_supported_when_rho_below_threshold(self) -> None:
        v = evaluate_hypotheses(
            rho_active_cpwer=0.1, perm_p_active_cpwer=0.4,
            rho_active_halluc=0.1, perm_p_active_halluc=0.4,
            rho_num_halluc=0.1,
            mode_s_active_counts={22: 1, 30: 1},
            mode_s_num_counts={22: 2, 30: 1},
            partial_active_silence=0.05,
            partial_active_lengthratio=0.2,
            partial_active_silence_active_ge1=0.1,
        )
        self.assertEqual(v["H38a"]["verdict"], "NOT SUPPORTED")

    def test_h38b_supported_when_all_mode_s_counts_le_two(self) -> None:
        v = evaluate_hypotheses(
            rho_active_cpwer=0.5, perm_p_active_cpwer=0.001,
            rho_active_halluc=0.4, perm_p_active_halluc=0.01,
            rho_num_halluc=0.3,
            mode_s_active_counts={22: 1, 30: 2},
            mode_s_num_counts={22: 2, 30: 1},
            partial_active_silence=0.05,
            partial_active_lengthratio=0.2,
            partial_active_silence_active_ge1=0.1,
        )
        self.assertEqual(v["H38b"]["verdict"], "SUPPORTED")

    def test_h38b_not_supported_when_any_mode_s_count_above_two(self) -> None:
        v = evaluate_hypotheses(
            rho_active_cpwer=0.5, perm_p_active_cpwer=0.001,
            rho_active_halluc=0.4, perm_p_active_halluc=0.01,
            rho_num_halluc=0.3,
            mode_s_active_counts={22: 3, 30: 1},  # w22 has 3 active -> violates
            mode_s_num_counts={22: 2, 30: 1},
            partial_active_silence=0.05,
            partial_active_lengthratio=0.2,
            partial_active_silence_active_ge1=0.1,
        )
        self.assertEqual(v["H38b"]["verdict"], "NOT SUPPORTED")

    def test_h38c_supported_when_partial_below_threshold(self) -> None:
        v = evaluate_hypotheses(
            rho_active_cpwer=0.5, perm_p_active_cpwer=0.001,
            rho_active_halluc=0.4, perm_p_active_halluc=0.01,
            rho_num_halluc=0.3,
            mode_s_active_counts={22: 1, 30: 1},
            mode_s_num_counts={22: 2, 30: 1},
            partial_active_silence=0.05,  # |0.05| < 0.1 -> mediated
            partial_active_lengthratio=0.3,
            partial_active_silence_active_ge1=0.2,
        )
        self.assertEqual(v["H38c"]["verdict"], "SUPPORTED")
        self.assertTrue(v["H38c"]["supported_primary"])
        self.assertFalse(v["H38c"]["supported_lengthratio"])

    def test_h38c_not_supported_when_partial_above_threshold(self) -> None:
        v = evaluate_hypotheses(
            rho_active_cpwer=0.5, perm_p_active_cpwer=0.001,
            rho_active_halluc=0.4, perm_p_active_halluc=0.01,
            rho_num_halluc=0.3,
            mode_s_active_counts={22: 1, 30: 1},
            mode_s_num_counts={22: 2, 30: 1},
            partial_active_silence=0.5,  # |0.5| >= 0.1 -> not mediated
            partial_active_lengthratio=0.05,
            partial_active_silence_active_ge1=0.4,
        )
        self.assertEqual(v["H38c"]["verdict"], "NOT SUPPORTED")
        self.assertTrue(v["H38c"]["supported_lengthratio"])


if __name__ == "__main__":
    unittest.main()
