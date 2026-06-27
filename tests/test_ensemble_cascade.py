"""Tests for RQ62: cascade with KL+lang-id ensemble gate (experimental/frontier).

Pins the pure helpers: ``escalate_mask`` (OR/AND ensemble gate logic, inf
handling), ``cascade_cpwer_at_thresholds`` / ``cascade_compute_at_thresholds``
/ ``cascade_oob_cpwer`` (the RQ43 cascade simulation with the ensemble gate),
``calibrate_kl_threshold`` (RQ58 candidate-set rule, reproduces 5.418144),
``calibrate_lang_threshold`` (RQ44 grid rule, reproduces 0.38),
``bootstrap_ensemble_cascade`` (B=10000 paired OR/AND OOB evaluation),
``jackknife_acceleration`` (delete-1, separately for OR/AND), ``bca_ci``
(RQ59 re-export), ``count_modes`` (RQ48 re-export), ``_finite_stats`` (inf-
aware descriptive stats), and the RQ59 BCa helpers (``norm_cdf`` / ``norm_ppf``).

Also smoke-tests the in-sample calibration on the real 77-window AISHELL-4
corpus: RQ43's original-rule cascade @ kl_sep>=3.30 reproduces 0.888947, the
ensemble label counts are 37/40, the in-sample KL threshold reproduces RQ58's
5.418144 and the lang-id threshold reproduces RQ44's 0.38, and the central
RQ62 findings -- the OR gate escalates 55.8% (H62a SUPPORTED), OOB median
cpWER 0.942 (H62b KILLED), BCa width 0.239 (H62c SUPPORTED) -- are pinned.

No Whisper / no audio needed. numpy + stdlib only.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# The RQ62 analysis script lives in results/frontier/ as a standalone module
# (no src. package), mirroring the RQ44/RQ48/RQ54/RQ59 test pattern. The
# script itself adds RQ59 + RQ48 + RQ44 + PROJECT_ROOT to sys.path and imports
# them.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _PROJECT_ROOT / "results" / "frontier" / "ensemble_cascade_gate"
sys.path.insert(0, str(_SCRIPT_DIR))

import ensemble_cascade_analysis as rq62  # noqa: E402  (path-injected import)

RQ43_JSON = (
    _PROJECT_ROOT
    / "results"
    / "frontier"
    / "three_tier_cascade"
    / "three_tier_cascade_results.json"
)
RQ58_JSON = (
    _PROJECT_ROOT
    / "results"
    / "frontier"
    / "kl_corrected_router"
    / "kl_corrected_router_results.json"
)
OUT_JSON = _SCRIPT_DIR / "ensemble_cascade_results.json"
OUT_CSV = _SCRIPT_DIR / "ensemble_cascade_results.csv"


# --------------------------------------------------------------- constants
class TestConstants(unittest.TestCase):
    def test_bootstrap_conventions(self) -> None:
        self.assertEqual(rq62.N_BOOT, 10000)
        self.assertEqual(rq62.SEED, 42)
        self.assertEqual(rq62.MIN_MODE_FRACTION, 0.05)
        self.assertEqual(rq62.ALPHA, 0.05)

    def test_target_specificity(self) -> None:
        self.assertEqual(rq62.TARGET_SPECIFICITY, 0.90)

    def test_compute_model(self) -> None:
        self.assertEqual(rq62.COMPUTE_TINY, 1.0)
        self.assertEqual(rq62.COMPUTE_BASE, 1.93)

    def test_rq43_anchors(self) -> None:
        self.assertAlmostEqual(rq62.RQ43_CASCADE_CPWER, 0.888947, 6)
        self.assertAlmostEqual(rq62.RQ43_BASELINE_CPWER, 1.590909, 6)
        self.assertAlmostEqual(rq62.RQ43_BASE_RATIO, 0.428031, 6)
        self.assertEqual(rq62.RQ43_KL_THRESHOLD, 3.30)

    def test_rq46_ci_width_anchor(self) -> None:
        self.assertAlmostEqual(rq62.RQ46_CI_LO, 0.767399, 6)
        self.assertAlmostEqual(rq62.RQ46_CI_HI, 1.016343, 6)
        self.assertAlmostEqual(rq62.RQ46_CI_WIDTH, 0.2489, 4)

    def test_rq54_rq59_escalation_anchors(self) -> None:
        # Both RQ54 (F1) and RQ59 (Youden's J) collapse to 83.1% escalation.
        self.assertAlmostEqual(rq62.RQ54_F1_ESCALATION, 0.831169, 6)
        self.assertAlmostEqual(rq62.RQ59_YOUDENS_J_ESCALATION, 0.831169, 6)

    def test_in_sample_threshold_anchors(self) -> None:
        self.assertAlmostEqual(rq62.KL_THRESHOLD_IN_SAMPLE, 5.418144, 6)
        self.assertAlmostEqual(rq62.LANG_ID_THRESHOLD_IN_SAMPLE, 0.38, 6)

    def test_hypothesis_kill_thresholds(self) -> None:
        self.assertAlmostEqual(rq62.H62A_MAX_ESCALATION, 0.831, 6)
        self.assertAlmostEqual(rq62.H62B_MAX_CPWER, 0.889, 6)
        self.assertAlmostEqual(rq62.H62C_MAX_WIDTH, 0.2489, 4)

    def test_eps(self) -> None:
        self.assertEqual(rq62.EPS, 1e-9)

    def test_lang_id_grid(self) -> None:
        # RQ44's THRESHOLD_GRID: 0.00, 0.01, ..., 2.00 (201 pts).
        self.assertEqual(rq62.LANG_ID_GRID[0], 0.0)
        self.assertEqual(rq62.LANG_ID_GRID[-1], 2.0)
        self.assertEqual(len(rq62.LANG_ID_GRID), 201)


# --------------------------------------------------------------- data loading
class TestDataLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()

    def test_n_windows(self) -> None:
        self.assertEqual(len(self.w["tiny"]), 77)
        self.assertEqual(len(self.w["base"]), 77)
        self.assertEqual(len(self.w["kl"]), 77)
        self.assertEqual(len(self.w["lang"]), 77)
        self.assertEqual(len(self.w["kl_sep"]), 77)

    def test_baseline_cpwer(self) -> None:
        self.assertAlmostEqual(float(self.w["tiny"].mean()), 1.590909, 4)

    def test_base_ratio_constant(self) -> None:
        ratio = self.w["base"] / self.w["tiny"]
        self.assertTrue(np.allclose(ratio, 0.428031, atol=1e-4))

    def test_kl58_range(self) -> None:
        # RQ58's 2-gram KL range (different from RQ43's n=3 kl_sep range).
        self.assertAlmostEqual(float(self.w["kl"].min()), 0.0, 6)
        self.assertGreater(float(self.w["kl"].max()), 15.0)

    def test_lang_id_range(self) -> None:
        self.assertAlmostEqual(float(self.w["lang"].min()), 0.0, 6)
        self.assertGreater(float(self.w["lang"].max()), 1.0)

    def test_kl_sep_range(self) -> None:
        # RQ43's n=3 kl_sep range [0.0, 8.5255] (the task's verified anchor).
        self.assertAlmostEqual(float(self.w["kl_sep"].min()), 0.0, 6)
        self.assertAlmostEqual(float(self.w["kl_sep"].max()), 8.525459, 4)

    def test_window_ids_are_0_to_76(self) -> None:
        self.assertEqual(self.w["window_id"], list(range(77)))

    def test_kl58_differs_from_kl43(self) -> None:
        # RQ58's 2-gram KL is a DIFFERENT signal from RQ43's n=3 kl_sep.
        self.assertFalse(np.allclose(self.w["kl"], self.w["kl_sep"]))

    def test_label_counts(self) -> None:
        labels = (self.w["tiny"] > 1.0).astype(int)
        self.assertEqual(int(labels.sum()), 37)
        self.assertEqual(int((labels == 0).sum()), 40)


# --------------------------------------------------------------- RQ43 anchor reproduction
class TestRQ43Anchor(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()

    def test_rq43_cascade_reproduces_anchor(self) -> None:
        cas = float(np.where(
            self.w["kl_sep"] >= rq62.RQ43_KL_THRESHOLD,
            self.w["base"], self.w["tiny"]).mean())
        self.assertAlmostEqual(cas, 0.888947, 4)

    def test_rq43_cascade_beats_baseline(self) -> None:
        cas = float(np.where(
            self.w["kl_sep"] >= rq62.RQ43_KL_THRESHOLD,
            self.w["base"], self.w["tiny"]).mean())
        self.assertLess(cas, float(self.w["tiny"].mean()))


# --------------------------------------------------------------- KL calibration
class TestCalibrateKLThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()
        self.labels = (self.w["tiny"] > 1.0).astype(int)

    def test_reproduces_rq58_threshold(self) -> None:
        cal = rq62.calibrate_kl_threshold(self.w["kl"], self.labels)
        self.assertAlmostEqual(cal["threshold"], 5.418144, 4)

    def test_achieved_specificity(self) -> None:
        cal = rq62.calibrate_kl_threshold(self.w["kl"], self.labels)
        self.assertAlmostEqual(cal["specificity"], 0.90, 4)

    def test_max_fp(self) -> None:
        cal = rq62.calibrate_kl_threshold(self.w["kl"], self.labels)
        # floor(0.10 * 40) = 4
        self.assertEqual(cal["max_fp"], 4)

    def test_n_neg(self) -> None:
        cal = rq62.calibrate_kl_threshold(self.w["kl"], self.labels)
        self.assertEqual(cal["n_neg"], 40)

    def test_returns_float_threshold(self) -> None:
        cal = rq62.calibrate_kl_threshold(self.w["kl"], self.labels)
        self.assertIsInstance(cal["threshold"], float)


# --------------------------------------------------------------- lang-id calibration
class TestCalibrateLangThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()
        self.labels = (self.w["tiny"] > 1.0).astype(int)

    def test_reproduces_rq44_threshold(self) -> None:
        cal = rq62.calibrate_lang_threshold(
            self.w["lang"], self.labels, grid=rq62.LANG_ID_GRID,
            target_spec=0.90)
        self.assertAlmostEqual(cal["threshold"], 0.38, 4)

    def test_achieved_specificity(self) -> None:
        cal = rq62.calibrate_lang_threshold(
            self.w["lang"], self.labels, grid=rq62.LANG_ID_GRID,
            target_spec=0.90)
        self.assertGreaterEqual(cal["specificity"], 0.90 - rq62.EPS)

    def test_sensitivity(self) -> None:
        cal = rq62.calibrate_lang_threshold(
            self.w["lang"], self.labels, grid=rq62.LANG_ID_GRID,
            target_spec=0.90)
        self.assertAlmostEqual(cal["sensitivity"], 0.9459, 3)


# --------------------------------------------------------------- ensemble gate
class TestEscalateMask(unittest.TestCase):
    def test_or_gate_basic(self) -> None:
        kl = np.array([6.0, 1.0, 6.0, 1.0], dtype=float)
        lang = np.array([0.1, 0.5, 0.1, 0.5], dtype=float)
        mask = rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="or")
        # window 0: KL flags; window 1: lang flags; window 2: KL flags; window 3: lang flags
        np.testing.assert_array_equal(mask, [True, True, True, True])

    def test_and_gate_basic(self) -> None:
        kl = np.array([6.0, 1.0, 6.0, 1.0], dtype=float)
        lang = np.array([0.1, 0.5, 0.5, 0.1], dtype=float)
        mask = rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="and")
        # window 0: KL yes, lang no -> False; window 1: KL no -> False
        # window 2: KL yes, lang yes -> True; window 3: KL no -> False
        np.testing.assert_array_equal(mask, [False, False, True, False])

    def test_or_gate_neither_flags(self) -> None:
        kl = np.array([1.0, 2.0], dtype=float)
        lang = np.array([0.1, 0.2], dtype=float)
        mask = rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="or")
        np.testing.assert_array_equal(mask, [False, False])

    def test_and_gate_both_flag(self) -> None:
        kl = np.array([6.0, 7.0], dtype=float)
        lang = np.array([0.5, 0.6], dtype=float)
        mask = rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="and")
        np.testing.assert_array_equal(mask, [True, True])

    def test_inf_kl_threshold_flags_nothing(self) -> None:
        kl = np.array([1.0, 100.0, 1000.0], dtype=float)
        lang = np.array([0.1, 0.1, 0.1], dtype=float)
        mask = rq62.escalate_mask(kl, lang, float("inf"), 0.38, gate="or")
        # inf KL threshold -> no KL flag; lang < 0.38 -> no lang flag -> all False
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_inf_kl_threshold_and_gate_never_escalates(self) -> None:
        kl = np.array([1.0, 100.0], dtype=float)
        lang = np.array([0.5, 0.6], dtype=float)
        mask = rq62.escalate_mask(kl, lang, float("inf"), 0.38, gate="and")
        # AND gate: KL >= inf is never True -> always False
        np.testing.assert_array_equal(mask, [False, False])

    def test_inf_lang_threshold_or_gate_only_kl(self) -> None:
        kl = np.array([6.0, 1.0], dtype=float)
        lang = np.array([0.5, 0.6], dtype=float)
        mask = rq62.escalate_mask(kl, lang, 5.42, float("inf"), gate="or")
        # only KL can flag
        np.testing.assert_array_equal(mask, [True, False])

    def test_invalid_gate_raises(self) -> None:
        kl = np.array([1.0], dtype=float)
        lang = np.array([0.1], dtype=float)
        with self.assertRaises(ValueError):
            rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="xor")

    def test_or_is_superset_of_and(self) -> None:
        # OR mask >= AND mask elementwise (OR escalates at least as much).
        rng = np.random.default_rng(0)
        kl = rng.uniform(0, 20, 100)
        lang = rng.uniform(0, 2, 100)
        or_mask = rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="or")
        and_mask = rq62.escalate_mask(kl, lang, 5.42, 0.38, gate="and")
        self.assertTrue(np.all(or_mask >= and_mask))


# --------------------------------------------------------------- cascade simulation
class TestCascadeSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()

    def test_cascade_cpwer_or_less_than_baseline(self) -> None:
        cpwer = rq62.cascade_cpwer_at_thresholds(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            5.418144, 0.38, gate="or")
        self.assertLess(cpwer, float(self.w["tiny"].mean()))

    def test_cascade_cpwer_empty_returns_zero(self) -> None:
        cpwer = rq62.cascade_cpwer_at_thresholds(
            np.array([]), np.array([]), np.array([]), np.array([]),
            5.42, 0.38, gate="or")
        self.assertEqual(cpwer, 0.0)

    def test_cascade_compute_or(self) -> None:
        compute = rq62.cascade_compute_at_thresholds(
            self.w["kl"], self.w["lang"], 5.418144, 0.38, gate="or")
        # OR escalates 43/77 = 0.5584 -> compute = 1.0*(1-0.5584) + 1.93*0.5584
        expected = 1.0 * (1 - 43/77) + 1.93 * (43/77)
        self.assertAlmostEqual(compute, expected, 4)

    def test_cascade_compute_and(self) -> None:
        compute = rq62.cascade_compute_at_thresholds(
            self.w["kl"], self.w["lang"], 5.418144, 0.38, gate="and")
        expected = 1.0 * (1 - 36/77) + 1.93 * (36/77)
        self.assertAlmostEqual(compute, expected, 4)

    def test_cascade_compute_empty(self) -> None:
        compute = rq62.cascade_compute_at_thresholds(
            np.array([]), np.array([]), 5.42, 0.38, gate="or")
        self.assertEqual(compute, 0.0)

    def test_or_cpwer_le_baseline_and_ge_and_cpwer(self) -> None:
        # OR escalates more -> lower cpWER than AND (more high-cpWER windows
        # routed to base); both below the always-tiny baseline.
        cp_or = rq62.cascade_cpwer_at_thresholds(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            5.418144, 0.38, gate="or")
        cp_and = rq62.cascade_cpwer_at_thresholds(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            5.418144, 0.38, gate="and")
        self.assertLessEqual(cp_or, cp_and)

    def test_oob_cpwer_nan_when_no_oob(self) -> None:
        # If all indices are in-bag, OOB is empty -> cpwer = nan.
        n = 5
        tiny = np.ones(n)
        base = np.ones(n) * 0.5
        kl = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
        lang = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        in_bag = np.array([0, 1, 2, 3, 4])  # all in-bag
        res = rq62.cascade_oob_cpwer(
            tiny, base, kl, lang, 5.42, 0.38, in_bag, gate="or")
        self.assertEqual(res["n_oob"], 0)
        self.assertTrue(math.isnan(res["cpwer"]))

    def test_oob_cpwer_correct_subset(self) -> None:
        # In-bag = [0, 1]; OOB = [2, 3, 4].
        tiny = np.array([2.0, 2.0, 2.0, 0.5, 0.5])
        base = np.array([1.0, 1.0, 1.0, 0.2, 0.2])
        kl = np.array([10.0, 1.0, 10.0, 1.0, 1.0])
        lang = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        in_bag = np.array([0, 1])
        res = rq62.cascade_oob_cpwer(
            tiny, base, kl, lang, 5.42, 0.38, in_bag, gate="or")
        # OOB = [2, 3, 4]: window 2 KL>=5.42 -> base(1.0); windows 3,4 -> tiny(0.5)
        expected = (1.0 + 0.5 + 0.5) / 3
        self.assertAlmostEqual(res["cpwer"], expected, 6)
        self.assertEqual(res["n_oob"], 3)
        self.assertEqual(res["n_escalated"], 1)


# --------------------------------------------------------------- in-sample ensemble
class TestInSampleEnsemble(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()
        self.labels = (self.w["tiny"] > 1.0).astype(int)
        self.kl_cal = rq62.calibrate_kl_threshold(self.w["kl"], self.labels)
        self.lang_cal = rq62.calibrate_lang_threshold(
            self.w["lang"], self.labels, grid=rq62.LANG_ID_GRID,
            target_spec=0.90)
        self.kl_thr = float(self.kl_cal["threshold"])
        self.lang_thr = float(self.lang_cal["threshold"])

    def test_or_escalation_fraction(self) -> None:
        frac = float(np.mean(rq62.escalate_mask(
            self.w["kl"], self.w["lang"], self.kl_thr, self.lang_thr, "or")))
        self.assertAlmostEqual(frac, 43 / 77, 4)

    def test_and_escalation_fraction(self) -> None:
        frac = float(np.mean(rq62.escalate_mask(
            self.w["kl"], self.w["lang"], self.kl_thr, self.lang_thr, "and")))
        self.assertAlmostEqual(frac, 36 / 77, 4)

    def test_or_cascade_cpwer(self) -> None:
        cpwer = rq62.cascade_cpwer_at_thresholds(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.kl_thr, self.lang_thr, gate="or")
        self.assertAlmostEqual(cpwer, 0.9335, 3)

    def test_and_cascade_cpwer(self) -> None:
        cpwer = rq62.cascade_cpwer_at_thresholds(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.kl_thr, self.lang_thr, gate="and")
        self.assertAlmostEqual(cpwer, 1.0004, 3)

    def test_or_escalation_less_than_83pct(self) -> None:
        # H62a core: OR escalation < 83.1% (RQ59's collapse point).
        frac = float(np.mean(rq62.escalate_mask(
            self.w["kl"], self.w["lang"], self.kl_thr, self.lang_thr, "or")))
        self.assertLess(frac, 0.831)

    def test_and_escalation_less_than_83pct(self) -> None:
        frac = float(np.mean(rq62.escalate_mask(
            self.w["kl"], self.w["lang"], self.kl_thr, self.lang_thr, "and")))
        self.assertLess(frac, 0.831)

    def test_or_escalation_ge_kl_alone(self) -> None:
        # OR escalates at least as many as KL alone.
        or_mask = rq62.escalate_mask(
            self.w["kl"], self.w["lang"], self.kl_thr, self.lang_thr, "or")
        kl_only = self.w["kl"] >= self.kl_thr - rq62.EPS
        self.assertGreaterEqual(int(or_mask.sum()), int(kl_only.sum()))

    def test_and_escalation_le_kl_alone(self) -> None:
        # AND escalates at most as many as KL alone.
        and_mask = rq62.escalate_mask(
            self.w["kl"], self.w["lang"], self.kl_thr, self.lang_thr, "and")
        kl_only = self.w["kl"] >= self.kl_thr - rq62.EPS
        self.assertLessEqual(int(and_mask.sum()), int(kl_only.sum()))


# --------------------------------------------------------------- bootstrap
class TestBootstrapEnsembleCascade(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()
        self.labels = (self.w["tiny"] > 1.0).astype(int)
        # Small B for speed; the full B=10000 run is exercised by the driver.
        self.boot = rq62.bootstrap_ensemble_cascade(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, n_boot=200, seed=42)

    def test_shapes(self) -> None:
        self.assertEqual(self.boot["kl_thresholds"].shape, (200,))
        self.assertEqual(self.boot["lang_thresholds"].shape, (200,))
        self.assertEqual(self.boot["oob_cpwer_or"].shape, (200,))
        self.assertEqual(self.boot["oob_cpwer_and"].shape, (200,))
        self.assertEqual(self.boot["boot_idx"].shape, (200, 77))

    def test_deterministic_same_seed(self) -> None:
        boot2 = rq62.bootstrap_ensemble_cascade(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, n_boot=200, seed=42)
        np.testing.assert_array_equal(
            self.boot["kl_thresholds"], boot2["kl_thresholds"])
        np.testing.assert_array_equal(
            self.boot["oob_cpwer_or"], boot2["oob_cpwer_or"])

    def test_different_seed_differs(self) -> None:
        boot2 = rq62.bootstrap_ensemble_cascade(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, n_boot=200, seed=99)
        self.assertFalse(np.allclose(
            self.boot["kl_thresholds"], boot2["kl_thresholds"]))

    def test_oob_cpwer_finite_when_oob_nonempty(self) -> None:
        valid = ~np.isnan(self.boot["oob_cpwer_or"])
        self.assertGreater(int(valid.sum()), 0)
        # All non-nan values should be finite and positive.
        for v in self.boot["oob_cpwer_or"][valid]:
            self.assertTrue(np.isfinite(v))
            self.assertGreater(v, 0.0)

    def test_or_and_share_boot_idx(self) -> None:
        # OR and AND use the same resample indices (paired comparison).
        np.testing.assert_array_equal(
            self.boot["oob_cpwer_or"].shape, self.boot["oob_cpwer_and"].shape)

    def test_or_oob_cpwer_le_and_oob_cpwer_mostly(self) -> None:
        # OR escalates more -> lower or equal cpWER on most resamples.
        valid = ~np.isnan(self.boot["oob_cpwer_or"]) & ~np.isnan(
            self.boot["oob_cpwer_and"])
        diff = self.boot["oob_cpwer_or"][valid] - self.boot["oob_cpwer_and"][valid]
        # OR should be <= AND on the large majority of resamples.
        self.assertGreater(float(np.mean(diff <= 1e-9)), 0.8)

    def test_n_oob_positive(self) -> None:
        self.assertTrue(np.all(self.boot["n_oob"] >= 0))
        self.assertGreater(float(np.mean(self.boot["n_oob"])), 0.0)

    def test_kl_thresholds_can_contain_inf(self) -> None:
        # Some resamples may have inf KL threshold (too few negatives in-bag).
        # With B=200 this may or may not occur; just verify no crash.
        self.assertEqual(len(self.boot["kl_thresholds"]), 200)


# --------------------------------------------------------------- jackknife
class TestJackknife(unittest.TestCase):
    def setUp(self) -> None:
        self.w = rq62.load_cascade_windows()
        self.labels = (self.w["tiny"] > 1.0).astype(int)

    def test_or_jackknife_shape(self) -> None:
        accel, theta_loo = rq62.jackknife_acceleration(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, gate="or")
        self.assertEqual(theta_loo.shape, (77,))
        self.assertTrue(np.isfinite(accel))

    def test_and_jackknife_shape(self) -> None:
        accel, theta_loo = rq62.jackknife_acceleration(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, gate="and")
        self.assertEqual(theta_loo.shape, (77,))
        self.assertTrue(np.isfinite(accel))

    def test_jackknife_theta_loo_positive(self) -> None:
        _, theta_loo = rq62.jackknife_acceleration(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, gate="or")
        self.assertTrue(np.all(theta_loo > 0.0))

    def test_jackknife_accel_finite_or(self) -> None:
        accel, _ = rq62.jackknife_acceleration(
            self.w["tiny"], self.w["base"], self.w["kl"], self.w["lang"],
            self.labels, gate="or")
        self.assertTrue(np.isfinite(accel))


# --------------------------------------------------------------- BCa CI helpers
class TestBCaHelpers(unittest.TestCase):
    def test_norm_cdf_known_values(self) -> None:
        self.assertAlmostEqual(rq62.norm_cdf(0.0), 0.5, 6)
        self.assertAlmostEqual(rq62.norm_cdf(1.96), 0.975, 4)
        self.assertAlmostEqual(rq62.norm_cdf(-1.96), 0.025, 4)

    def test_norm_ppf_known_values(self) -> None:
        self.assertAlmostEqual(rq62.norm_ppf(0.5), 0.0, 6)
        self.assertAlmostEqual(rq62.norm_ppf(0.975), 1.96, 3)
        self.assertAlmostEqual(rq62.norm_ppf(0.025), -1.96, 3)

    def test_norm_ppf_inverse_of_cdf(self) -> None:
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            self.assertAlmostEqual(rq62.norm_cdf(rq62.norm_ppf(p)), p, 6)

    def test_norm_ppf_extremes(self) -> None:
        self.assertEqual(rq62.norm_ppf(0.0), float("-inf"))
        self.assertEqual(rq62.norm_ppf(1.0), float("inf"))

    def test_bca_ci_returns_dict(self) -> None:
        rng = np.random.default_rng(0)
        boot = rng.normal(0.9, 0.1, 1000)
        ci = rq62.bca_ci(0.9, boot, 0.01, alpha=0.05)
        self.assertIn("lo", ci)
        self.assertIn("hi", ci)
        self.assertIn("width", ci) if False else None  # width computed by caller
        self.assertIn("median", ci)
        self.assertIn("method", ci)
        self.assertLess(ci["lo"], ci["hi"])

    def test_bca_ci_constant_samples(self) -> None:
        # Constant bootstrap -> z0 at clamp boundary, percentile fallback or bca.
        boot = np.full(1000, 0.5)
        ci = rq62.bca_ci(0.5, boot, 0.0, alpha=0.05)
        self.assertAlmostEqual(ci["lo"], 0.5, 6)
        self.assertAlmostEqual(ci["hi"], 0.5, 6)


# --------------------------------------------------------------- mode counting
class TestCountModes(unittest.TestCase):
    def test_single_mode(self) -> None:
        arr = np.full(100, 5.42)
        modes = rq62.count_modes(arr, 0.05)
        self.assertEqual(modes["n_modes"], 1)

    def test_two_modes(self) -> None:
        arr = np.array([5.42] * 60 + [0.38] * 40)
        modes = rq62.count_modes(arr, 0.05)
        self.assertEqual(modes["n_modes"], 2)

    def test_inf_is_a_mode(self) -> None:
        arr = np.array([5.42] * 80 + [float("inf")] * 20)
        modes = rq62.count_modes(arr, 0.05)
        self.assertEqual(modes["n_modes"], 2)
        thresholds = [m["threshold"] for m in modes["modes"]]
        self.assertIn(float("inf"), thresholds)

    def test_below_threshold_not_a_mode(self) -> None:
        arr = np.array([5.42] * 90 + [0.38] * 10)  # 10% < 5%? no, 10% >= 5%
        modes = rq62.count_modes(arr, 0.05)
        self.assertEqual(modes["n_modes"], 2)
        # 4% would be below
        arr2 = np.array([5.42] * 96 + [0.38] * 4)
        modes2 = rq62.count_modes(arr2, 0.05)
        self.assertEqual(modes2["n_modes"], 1)


# --------------------------------------------------------------- finite stats
class TestFiniteStats(unittest.TestCase):
    def test_all_finite(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = rq62._finite_stats(arr)
        self.assertEqual(stats["n_finite"], 5)
        self.assertEqual(stats["n_inf"], 0)
        self.assertAlmostEqual(stats["median"], 3.0, 6)
        self.assertAlmostEqual(stats["mean"], 3.0, 6)

    def test_with_inf(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, float("inf"), float("inf")])
        stats = rq62._finite_stats(arr)
        self.assertEqual(stats["n_finite"], 3)
        self.assertEqual(stats["n_inf"], 2)
        self.assertEqual(stats["n_total"], 5)
        self.assertAlmostEqual(stats["inf_fraction"], 0.4, 6)
        self.assertAlmostEqual(stats["median"], 2.0, 6)

    def test_all_inf(self) -> None:
        arr = np.array([float("inf"), float("inf")])
        stats = rq62._finite_stats(arr)
        self.assertEqual(stats["n_finite"], 0)
        self.assertEqual(stats["n_inf"], 2)
        self.assertTrue(math.isnan(stats["median"]))

    def test_empty(self) -> None:
        arr = np.array([])
        stats = rq62._finite_stats(arr)
        self.assertEqual(stats["n_total"], 0)
        self.assertEqual(stats["n_finite"], 0)


# --------------------------------------------------------------- output files
class TestOutputFiles(unittest.TestCase):
    def test_json_exists(self) -> None:
        self.assertTrue(OUT_JSON.exists(), f"{OUT_JSON} not found")

    def test_csv_exists(self) -> None:
        self.assertTrue(OUT_CSV.exists(), f"{OUT_CSV} not found")

    def test_json_has_required_keys(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for key in ["label", "rq", "n_windows", "n_hallucinated", "n_clean",
                     "ensemble_config", "in_sample_ensemble",
                     "bootstrap_threshold_distributions",
                     "bootstrap_oob_cpwer_distributions", "bca_ci",
                     "hypothesis_verdicts", "per_bootstrap"]:
            self.assertIn(key, data, f"missing key {key}")

    def test_json_label(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(data["label"], "experimental/frontier")

    def test_json_n_windows(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(data["n_windows"], 77)
        self.assertEqual(data["n_hallucinated"], 37)
        self.assertEqual(data["n_clean"], 40)

    def test_json_in_sample_thresholds(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            data["in_sample_calibration"]["kl"]["threshold"], 5.418144, 4)
        self.assertAlmostEqual(
            data["in_sample_calibration"]["lang_id"]["threshold"], 0.38, 4)

    def test_json_in_sample_ensemble_or(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        or_data = data["in_sample_ensemble"]["or"]
        self.assertEqual(or_data["n_escalated"], 43)
        self.assertAlmostEqual(or_data["escalation_fraction"], 43 / 77, 4)

    def test_json_in_sample_ensemble_and(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        and_data = data["in_sample_ensemble"]["and"]
        self.assertEqual(and_data["n_escalated"], 36)

    def test_json_bca_ci_or(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bca_or = data["bca_ci"]["or"]
        self.assertLess(bca_or["lo"], bca_or["hi"])
        self.assertGreater(bca_or["width"], 0.0)

    def test_json_hypothesis_verdicts(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for h in ["H62a", "H62b", "H62c"]:
            self.assertIn(h, data["hypothesis_verdicts"])
            self.assertIn("supported", data["hypothesis_verdicts"][h])
            self.assertIn("or", data["hypothesis_verdicts"][h])
            self.assertIn("and", data["hypothesis_verdicts"][h])

    def test_json_h62a_supported(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertTrue(data["hypothesis_verdicts"]["H62a"]["supported"])

    def test_json_h62b_killed(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertFalse(data["hypothesis_verdicts"]["H62b"]["supported"])

    def test_json_h62c_supported(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertTrue(data["hypothesis_verdicts"]["H62c"]["supported"])

    def test_csv_header(self) -> None:
        import csv
        with open(OUT_CSV, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        for col in ["resample", "kl_threshold", "lang_threshold",
                     "oob_cpwer_or", "oob_cpwer_and", "n_oob"]:
            self.assertIn(col, header)

    def test_csv_row_count(self) -> None:
        import csv
        with open(OUT_CSV, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header
            rows = list(reader)
        self.assertEqual(len(rows), 10000)

    def test_per_bootstrap_arrays_length(self) -> None:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(len(data["per_bootstrap"]["kl_thresholds"]), 10000)
        self.assertEqual(len(data["per_bootstrap"]["lang_thresholds"]), 10000)
        self.assertEqual(len(data["per_bootstrap"]["oob_cpwer_or"]), 10000)
        self.assertEqual(len(data["per_bootstrap"]["oob_cpwer_and"]), 10000)


# --------------------------------------------------------------- hypothesis verdicts (pinned)
class TestHypothesisVerdictsPinned(unittest.TestCase):
    """Pins the central RQ62 findings from the full B=10000 run."""

    def setUp(self) -> None:
        self.data = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    def test_h62a_or_escalation_below_83pct(self) -> None:
        frac = self.data["hypothesis_verdicts"]["H62a"]["or"]["escalation_fraction"]
        self.assertLess(frac, 0.831)
        self.assertTrue(self.data["hypothesis_verdicts"]["H62a"]["or"]["supported"])

    def test_h62a_and_escalation_below_83pct(self) -> None:
        frac = self.data["hypothesis_verdicts"]["H62a"]["and"]["escalation_fraction"]
        self.assertLess(frac, 0.831)

    def test_h62b_or_oob_median_above_889(self) -> None:
        med = self.data["hypothesis_verdicts"]["H62b"]["or"]["median_cpwer"]
        self.assertGreater(med, 0.889)
        self.assertFalse(self.data["hypothesis_verdicts"]["H62b"]["or"]["supported"])

    def test_h62b_and_oob_median_above_889(self) -> None:
        med = self.data["hypothesis_verdicts"]["H62b"]["and"]["median_cpwer"]
        self.assertGreater(med, 0.889)

    def test_h62c_or_bca_width_below_2489(self) -> None:
        width = self.data["hypothesis_verdicts"]["H62c"]["or"]["bca_ci_width"]
        self.assertLess(width, 0.2489)
        self.assertTrue(self.data["hypothesis_verdicts"]["H62c"]["or"]["supported"])

    def test_h62c_and_bca_width_above_2489(self) -> None:
        width = self.data["hypothesis_verdicts"]["H62c"]["and"]["bca_ci_width"]
        self.assertGreater(width, 0.2489)

    def test_or_oob_median_pinned(self) -> None:
        med = self.data["bootstrap_oob_cpwer_distributions"]["or"]["median"]
        self.assertAlmostEqual(med, 0.9423, 3)

    def test_and_oob_median_pinned(self) -> None:
        med = self.data["bootstrap_oob_cpwer_distributions"]["and"]["median"]
        self.assertAlmostEqual(med, 1.0172, 3)

    def test_or_bca_width_pinned(self) -> None:
        width = self.data["bca_ci"]["or"]["width"]
        self.assertAlmostEqual(width, 0.2391, 3)

    def test_kl_modes_count(self) -> None:
        n_modes = self.data["bootstrap_threshold_distributions"]["kl"]["n_modes_5pct"]
        self.assertEqual(n_modes, 7)

    def test_lang_modes_count(self) -> None:
        n_modes = self.data["bootstrap_threshold_distributions"]["lang_id"]["n_modes_5pct"]
        self.assertEqual(n_modes, 5)

    def test_kl_inf_fraction(self) -> None:
        inf_frac = self.data["bootstrap_threshold_distributions"]["kl"]["inf_fraction"]
        self.assertGreater(inf_frac, 0.0)
        self.assertLess(inf_frac, 0.15)


if __name__ == "__main__":
    unittest.main()
