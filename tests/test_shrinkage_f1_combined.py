"""Tests for RQ66: shrinkage + F1 combined threshold calibration
(experimental/frontier).

Pins the pure helpers: ``shrinkage_f1_objective``, ``calibrate_shrinkage_f1``,
``calibrate_shrinkage_f1_beta22``, ``hartigans_dip``, ``_summarise_lambda``,
``select_best_lambda``, and the hypothesis helpers ``_h66a_supported`` /
``_h66b_supported`` / ``_h66c_supported``. Also pins the module constants, the
λ=0 ↔ RQ48 ``calibrate_f1`` equivalence (in-sample AND on a small bootstrap
draw), the shrinkage "pull toward prior" effect on synthetic Mode-S-style
data, the verbatim reuse of RQ48's ``count_modes`` and RQ44's bootstrap
framework, and the hypothesis kill-conditions (H66a/b/c). A real-data smoke
test reproduces RQ44's in-sample 0.38 threshold and 1.043 corrected cpWER.

No Whisper / no audio / no LLM needed. numpy + stdlib only (scipy optional,
used only for Hartigan's dip test with a try/except guard).
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# The RQ66 analysis script lives in results/frontier/ as a standalone module
# (no src. package), mirroring the RQ44/RQ48/RQ54/RQ61 test pattern.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _PROJECT_ROOT / "results" / "frontier" / "shrinkage_f1_combined_calibration"
sys.path.insert(0, str(_SCRIPT_DIR))
import shrinkage_f1_combined_analysis as rq66  # noqa: E402  (path-injected import)

# RQ44's module is needed for the verbatim-reuse cross-check + bootstrap draw.
_RQ44_DIR = _PROJECT_ROOT / "results" / "frontier" / "bootstrap_threshold_stability"
sys.path.insert(0, str(_RQ44_DIR))
import bootstrap_threshold_analysis as rq44  # noqa: E402  (path-injected import)

# RQ48's module is needed for the F1-equivalence cross-check.
_RQ48_DIR = _PROJECT_ROOT / "results" / "frontier" / "calibration_rule_comparison"
sys.path.insert(0, str(_RQ48_DIR))
import calibration_rule_analysis as rq48  # noqa: E402  (path-injected import)

AISHELL4_JSON = (
    _PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)


def _load_real_signals():
    """Load the 77-window AISHELL-4 signals used by RQ44/RQ48/RQ61/RQ66."""
    data = json.loads(AISHELL4_JSON.read_text(encoding="utf-8"))
    windows = data["windows"]
    lang_ent = np.array([rq44.max_across_speakers(w) for w in windows], dtype=float)
    mixed = np.array([float(w["always_mixed_cpwer"]) for w in windows], dtype=float)
    sep = np.array([float(w["always_separated_cpwer"]) for w in windows], dtype=float)
    labels = (sep > rq44.CATASTROPHIC_CPWER).astype(int)
    return lang_ent, mixed, sep, labels


# ----------------------------------------------------------- constants
class TestConstants(unittest.TestCase):
    def test_bootstrap_conventions(self) -> None:
        self.assertEqual(rq66.N_BOOT, 1000)
        self.assertEqual(rq66.SEED, 42)
        self.assertEqual(rq66.MIN_MODE_FRACTION, 0.05)

    def test_prior_mean_matches_rq44_bootstrap_median(self) -> None:
        self.assertAlmostEqual(rq66.PRIOR_MEAN, 0.38, places=6)

    def test_lambdas_match_rq61_grid(self) -> None:
        self.assertEqual(rq66.LAMBDAS, [0.0, 0.01, 0.1, 0.5, 1.0])

    def test_threshold_grid_is_rq44_verbatim(self) -> None:
        self.assertIs(rq66.THRESHOLD_GRID, rq44.THRESHOLD_GRID)
        self.assertEqual(rq66.THRESHOLD_GRID[0], 0.0)
        self.assertEqual(rq66.THRESHOLD_GRID[-1], 2.0)
        self.assertEqual(len(rq66.THRESHOLD_GRID), 201)

    def test_eps_and_catastrophic_cpwer_are_rq44_verbatim(self) -> None:
        self.assertIs(rq66.EPS, rq44.EPS)
        self.assertEqual(rq66.EPS, 1e-9)
        self.assertIs(rq66.CATASTROPHIC_CPWER, rq44.CATASTROPHIC_CPWER)
        self.assertEqual(rq66.CATASTROPHIC_CPWER, 1.0)

    def test_dip_multimodal_threshold_matches_issue(self) -> None:
        self.assertEqual(rq66.DIP_MULTIMODAL_THRESHOLD, 0.05)

    def test_hypothesis_kill_thresholds(self) -> None:
        self.assertEqual(rq66.H66A_MAX_MODES, 2)
        self.assertEqual(rq66.H66B_MAX_CPWER, 1.056)
        self.assertEqual(rq66.H66C_MAX_WIDTH, 0.2489)

    def test_rq44_reference_anchors(self) -> None:
        self.assertEqual(rq66.RQ44_OOB_CPWER_MEDIAN, 1.056)
        self.assertEqual(rq66.RQ44_N_MODES_5PCT, 5)
        self.assertEqual(rq66.RQ44_INTERVAL_WIDTH, 0.94)

    def test_rq48_rq54_rq61_reference_anchors(self) -> None:
        self.assertEqual(rq66.RQ48_F1_MODES, 2)
        self.assertEqual(rq66.RQ61_SHRINKAGE_MODES, 3)
        self.assertEqual(rq66.RQ54_F1_CPWER_CI_WIDTH, 0.2489)


# ----------------------------------------------------------- shrinkage_f1_objective
class TestShrinkageF1Objective(unittest.TestCase):
    def test_objective_equals_f1_minus_penalty(self) -> None:
        obj = rq66.shrinkage_f1_objective(
            threshold=0.5, f1=0.9, prior_mean=0.38, lam=0.1
        )
        self.assertAlmostEqual(obj, 0.9 - 0.1 * abs(0.5 - 0.38), places=6)

    def test_lam_zero_objective_equals_f1(self) -> None:
        obj = rq66.shrinkage_f1_objective(0.95, 0.8, 0.38, 0.0)
        self.assertAlmostEqual(obj, 0.8, places=6)

    def test_threshold_at_prior_has_zero_penalty(self) -> None:
        obj = rq66.shrinkage_f1_objective(0.38, 0.7, 0.38, 1.0)
        self.assertAlmostEqual(obj, 0.7, places=6)

    def test_penalty_is_lam_times_abs_diff(self) -> None:
        # Two thresholds equidistant from prior (0.28 and 0.48) -> same penalty.
        o_lo = rq66.shrinkage_f1_objective(0.28, 0.9, 0.38, 0.5)
        o_hi = rq66.shrinkage_f1_objective(0.48, 0.9, 0.38, 0.5)
        self.assertAlmostEqual(o_lo, o_hi, places=6)

    def test_higher_lam_stronger_penalty(self) -> None:
        # Same threshold/f1; larger lambda -> smaller objective.
        o_small = rq66.shrinkage_f1_objective(0.95, 0.9, 0.38, 0.1)
        o_large = rq66.shrinkage_f1_objective(0.95, 0.9, 0.38, 1.0)
        self.assertGreater(o_small, o_large)

    def test_farther_threshold_smaller_objective(self) -> None:
        # Same f1; threshold farther from prior -> smaller objective.
        o_near = rq66.shrinkage_f1_objective(0.40, 0.9, 0.38, 0.5)
        o_far = rq66.shrinkage_f1_objective(0.95, 0.9, 0.38, 0.5)
        self.assertGreater(o_near, o_far)

    def test_f1_zero_means_negative_penalty_dominates(self) -> None:
        # f1=0 at threshold far from prior -> negative objective.
        obj = rq66.shrinkage_f1_objective(0.95, 0.0, 0.38, 0.5)
        self.assertLess(obj, 0.0)


# ----------------------------------------------------------- calibrate_shrinkage_f1
class TestCalibrateShrinkageF1(unittest.TestCase):
    def test_separable_case_maximises_f1(self) -> None:
        # negs 0/0.1/0.2, pos 1.0/1.1. F1=1.0 (sens=1, spec=1) for t in (0.2, 1.0];
        # tie-break -> lowest threshold in tie = 0.21.
        scores = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
        labels = np.array([0, 0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        self.assertAlmostEqual(out["f1"], 1.0, places=6)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["specificity"], 1.0)

    def test_returns_all_confusion_counts(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        for key in ("threshold", "sensitivity", "specificity", "precision",
                    "f1", "tp", "fp", "tn", "fn", "objective", "penalty",
                    "lambda", "prior_mean"):
            self.assertIn(key, out)
        self.assertEqual(out["tp"] + out["fn"], 2)
        self.assertEqual(out["fp"] + out["tn"], 2)

    def test_objective_field_matches_f1_minus_penalty(self) -> None:
        scores = np.array([0.0, 0.4, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.5)
        expected = out["f1"] - 0.5 * abs(out["threshold"] - 0.38)
        self.assertAlmostEqual(out["objective"], expected, places=6)

    def test_penalty_field_matches_lam_times_abs_diff(self) -> None:
        scores = np.array([0.0, 0.4, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.5)
        self.assertAlmostEqual(
            out["penalty"], 0.5 * abs(out["threshold"] - 0.38), places=6
        )

    def test_lambda_field_stored(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        for lam in rq66.LAMBDAS:
            out = rq66.calibrate_shrinkage_f1(scores, labels, lam=lam)
            self.assertAlmostEqual(out["lambda"], lam, places=6)

    def test_empty_positives_safe(self) -> None:
        scores = np.array([0.0, 0.25, 0.5])
        labels = np.array([0, 0, 0])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.5)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertEqual(out["tp"], 0)
        self.assertEqual(out["fn"], 0)
        # F1 = 0 (no positives); objective = -penalty.
        self.assertAlmostEqual(out["f1"], 0.0, places=6)

    def test_empty_negatives_safe(self) -> None:
        scores = np.array([0.3, 0.6, 0.9])
        labels = np.array([1, 1, 1])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.5)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["specificity"], 1.0)
        self.assertEqual(out["fp"], 0)

    def test_default_prior_mean_is_038(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out_default = rq66.calibrate_shrinkage_f1(scores, labels, lam=1.0)
        out_explicit = rq66.calibrate_shrinkage_f1(
            scores, labels, lam=1.0, prior_mean=0.38
        )
        self.assertAlmostEqual(
            out_default["threshold"], out_explicit["threshold"], places=6
        )

    def test_default_grid_is_threshold_grid(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out_default = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        out_explicit = rq66.calibrate_shrinkage_f1(
            scores, labels, lam=0.0, grid=rq66.THRESHOLD_GRID
        )
        self.assertAlmostEqual(
            out_default["threshold"], out_explicit["threshold"], places=6
        )

    def test_f1_value_matches_2_prec_rec_over_sum(self) -> None:
        scores = np.array([0.0, 0.4, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        prec = out["tp"] / (out["tp"] + out["fp"]) if (out["tp"] + out["fp"]) > 0 else 0.0
        rec = out["sensitivity"]
        expected_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        self.assertAlmostEqual(out["f1"], expected_f1, places=6)


# ----------------------------------------------- lam=0 ↔ RQ48 calibrate_f1 equivalence
class TestLamZeroEquivalenceRQ48(unittest.TestCase):
    """lam=0 (no shrinkage) must reproduce RQ48's calibrate_f1 exactly: max F1
    with lowest-threshold tie-break."""

    def test_lam_zero_matches_rq48_on_real_data(self) -> None:
        lang_ent, _, _, labels = _load_real_signals()
        out_combined = rq66.calibrate_shrinkage_f1(lang_ent, labels, lam=0.0)
        out_rq48 = rq48.calibrate_f1(lang_ent, labels)
        self.assertAlmostEqual(
            out_combined["threshold"], out_rq48["threshold"], places=6)
        self.assertAlmostEqual(
            out_combined["sensitivity"], out_rq48["sensitivity"], places=6)
        self.assertAlmostEqual(
            out_combined["specificity"], out_rq48["specificity"], places=6)
        self.assertEqual(out_combined["tp"], out_rq48["tp"])
        self.assertEqual(out_combined["fp"], out_rq48["fp"])
        self.assertAlmostEqual(out_combined["f1"], out_rq48["f1"], places=6)

    def test_lam_zero_matches_rq48_real_threshold_038(self) -> None:
        lang_ent, _, _, labels = _load_real_signals()
        out = rq66.calibrate_shrinkage_f1(lang_ent, labels, lam=0.0)
        self.assertAlmostEqual(out["threshold"], 0.38, places=6)
        self.assertAlmostEqual(out["f1"], 0.933333, places=5)

    def test_lam_zero_matches_rq48_on_synthetic_separable(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
        labels = np.array([0, 0, 0, 1, 1])
        out_combined = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        out_rq48 = rq48.calibrate_f1(scores, labels)
        self.assertAlmostEqual(
            out_combined["threshold"], out_rq48["threshold"], places=6)

    def test_lam_zero_matches_rq48_on_synthetic_overlapping(self) -> None:
        # Overlapping scores where F1's precision/recall trade-off matters.
        scores = np.array([0.0, 0.35, 0.37, 0.39, 0.40, 0.42, 0.95, 1.0])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out_combined = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        out_rq48 = rq48.calibrate_f1(scores, labels)
        self.assertAlmostEqual(
            out_combined["threshold"], out_rq48["threshold"], places=6)
        self.assertAlmostEqual(
            out_combined["f1"], out_rq48["f1"], places=6)

    def test_lam_zero_matches_rq48_on_synthetic_tie_break(self) -> None:
        # All hall have very high entropy -> many thresholds tie on F1;
        # tie-break must match RQ48 (lowest threshold).
        scores = np.array([0.0, 0.1, 0.5, 0.6, 5.0, 5.0])
        labels = np.array([0, 0, 0, 0, 1, 1])
        out_combined = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        out_rq48 = rq48.calibrate_f1(scores, labels)
        self.assertAlmostEqual(
            out_combined["threshold"], out_rq48["threshold"], places=6)

    def test_lam_zero_matches_rq48_on_small_bootstrap(self) -> None:
        # Stronger: lam=0 matches RQ48 on every resample of a small bootstrap.
        lang_ent, _, _, labels = _load_real_signals()
        n = len(lang_ent)
        boot_idx = rq44.bootstrap_indices(n, 50, 42)  # B=50 (fast)
        for b in range(50):
            idx = boot_idx[b]
            out_combined = rq66.calibrate_shrinkage_f1(
                lang_ent[idx], labels[idx], lam=0.0)
            out_rq48 = rq48.calibrate_f1(lang_ent[idx], labels[idx])
            self.assertAlmostEqual(
                out_combined["threshold"], out_rq48["threshold"], places=6,
                msg=f"threshold mismatch at bootstrap b={b}")
            self.assertAlmostEqual(
                out_combined["f1"], out_rq48["f1"], places=6,
                msg=f"f1 mismatch at b={b}")


# --------------------------------------------------- shrinkage "pull" effect
class TestShrinkagePullTowardPrior(unittest.TestCase):
    """Shrinkage with lam>0 pulls the threshold toward the prior (0.38)."""

    def _mode_s_synthetic(self):
        # 2 low-entropy hallucinated (Mode S, score 0.02), 35 high-entropy hall
        # (5.0), 40 clean (0.0). At t=0.01: flags 0.02 & 5.0 -> sens=1.0, spec=1.0.
        # At t=0.38: flags only 5.0 -> sens=35/37, spec=1.0.
        scores = np.concatenate([
            np.full(2, 0.02), np.full(35, 5.0), np.zeros(40),
        ])
        labels = np.concatenate([np.ones(37), np.zeros(40)])
        return scores, labels

    def test_lam_zero_picks_low_threshold_mode_s(self) -> None:
        scores, labels = self._mode_s_synthetic()
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        # F1 at t=0.01: prec=1.0, rec=1.0 -> F1=1.0. F1 at t=0.38: prec=1.0,
        # rec=35/37 -> F1=0.972. So F1 picks 0.01 (lam=0).
        self.assertLess(out["threshold"], 0.1)
        self.assertAlmostEqual(out["f1"], 1.0, places=6)

    def test_large_lam_pulls_to_prior(self) -> None:
        scores, labels = self._mode_s_synthetic()
        out = rq66.calibrate_shrinkage_f1(scores, labels, lam=1.0)
        # Penalty at 0.01 = 1.0 * 0.37 = 0.37; objective = 1.0 - 0.37 = 0.63.
        # Penalty at 0.38 = 0.0; objective = 0.972 - 0.0 = 0.972. So 0.38 wins.
        self.assertAlmostEqual(out["threshold"], 0.38, places=6)

    def test_intermediate_lam_between_low_and_prior(self) -> None:
        scores, labels = self._mode_s_synthetic()
        out_lo = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        out_hi = rq66.calibrate_shrinkage_f1(scores, labels, lam=1.0)
        # Monotone pull: threshold(0.0) < threshold(1.0).
        self.assertLess(out_lo["threshold"], out_hi["threshold"])

    def test_shrinkage_pulls_high_threshold_toward_prior(self) -> None:
        # Construct a case where lam=0 picks a high threshold (F1-tied) but
        # shrinkage prefers the one closer to 0.38.
        # negs 0.0, 0.5; pos 0.6, 0.6, 5.0. F1 at t=0.51: prec=1.0, rec=1.0 -> 1.0.
        # F1 at t=0.38: prec=2/3 (0.5 clean flagged), rec=1.0 -> 0.8.
        # F1 at t=0.61: prec=1.0, rec=2/3 -> 0.8. So lam=0 picks 0.51 (F1=1.0).
        # Shrinkage: 0.51 penalty = lam*0.13; no other F1=1.0 threshold exists
        # closer to 0.38 (0.38 itself has F1=0.8). So shrinkage stays at 0.51
        # but the penalty reduces the objective. Verify threshold stays at 0.51.
        scores = np.array([0.0, 0.5, 0.6, 0.6, 5.0])
        labels = np.array([0, 0, 1, 1, 1])
        out_zero = rq66.calibrate_shrinkage_f1(scores, labels, lam=0.0)
        out_high = rq66.calibrate_shrinkage_f1(scores, labels, lam=1.0)
        self.assertAlmostEqual(out_zero["f1"], 1.0, places=6)
        # Both pick 0.51 (only F1=1.0 threshold); penalty differs.
        self.assertAlmostEqual(
            out_zero["threshold"], out_high["threshold"], places=6)
        self.assertLess(out_high["objective"], out_zero["objective"])

    def test_shrinkage_never_picks_below_zero(self) -> None:
        # All thresholds in grid are >= 0.0; verify shrinkage respects this.
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        for lam in rq66.LAMBDAS:
            out = rq66.calibrate_shrinkage_f1(scores, labels, lam=lam)
            self.assertGreaterEqual(out["threshold"], 0.0)


# --------------------------------------------------- Beta(2,2) variant
class TestCalibrateShrinkageF1Beta22(unittest.TestCase):
    def test_separable_case_picks_posterior_mode(self) -> None:
        # negs 0/0.1/0.2, pos 1.0/1.1. F1=1.0 for t in (0.2, 1.0]. Beta(2,2)
        # density = 6*t*(1-t), maximised at t=0.5. Posterior = F1 * prior,
        # so picks t closest to 0.5 within the F1=1.0 tie -> 0.5.
        scores = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
        labels = np.array([0, 0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1_beta22(scores, labels)
        self.assertAlmostEqual(out["threshold"], 0.5, places=6)
        self.assertAlmostEqual(out["f1"], 1.0, places=6)

    def test_returns_beta22_prior_metadata(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1_beta22(scores, labels)
        self.assertEqual(out["prior"], "Beta(2,2)")
        self.assertEqual(out["prior_mode"], 0.5)
        for key in ("threshold", "sensitivity", "specificity", "precision",
                    "f1", "tp", "fp", "tn", "fn", "posterior", "prior_density"):
            self.assertIn(key, out)

    def test_posterior_field_matches_f1_times_prior(self) -> None:
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1_beta22(scores, labels)
        t = out["threshold"]
        expected_prior = t * (1.0 - t) if 0.0 <= t <= 1.0 else 0.0
        expected_posterior = out["f1"] * expected_prior
        self.assertAlmostEqual(out["posterior"], expected_posterior, places=6)
        self.assertAlmostEqual(out["prior_density"], expected_prior, places=6)

    def test_threshold_in_unit_interval(self) -> None:
        # Beta(2,2) density is 0 outside [0, 1] -> posterior mode in [0, 1].
        scores = np.array([0.0, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        out = rq66.calibrate_shrinkage_f1_beta22(scores, labels)
        self.assertGreaterEqual(out["threshold"], 0.0)
        self.assertLessEqual(out["threshold"], 1.0)

    def test_empty_positives_safe(self) -> None:
        scores = np.array([0.0, 0.25, 0.5])
        labels = np.array([0, 0, 0])
        out = rq66.calibrate_shrinkage_f1_beta22(scores, labels)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertEqual(out["tp"], 0)
        self.assertEqual(out["f1"], 0.0)


# ----------------------------------------------------------- hartigans_dip
class TestHartigansDip(unittest.TestCase):
    def test_lt3_points_returns_trivial_unimodal(self) -> None:
        out = rq66.hartigans_dip(np.array([0.38, 0.40]))
        self.assertEqual(out["method"], "trivial_lt3")
        self.assertEqual(out["dip"], 0.0)
        self.assertFalse(out["multimodal"])

    def test_empty_array_returns_trivial(self) -> None:
        out = rq66.hartigans_dip(np.array([]))
        self.assertEqual(out["method"], "trivial_lt3")
        self.assertEqual(out["dip"], 0.0)

    def test_returns_dict_with_expected_keys(self) -> None:
        rng = np.random.default_rng(42)
        vals = rng.normal(0.38, 0.1, size=100)
        out = rq66.hartigans_dip(vals)
        for key in ("dip", "multimodal", "pvalue", "method"):
            self.assertIn(key, out)
        # Either scipy available (dip is a float) or unavailable (dip is None).
        if out["method"] == "scipy_unavailable":
            self.assertIsNone(out["dip"])
            self.assertIsNone(out["multimodal"])
        else:
            self.assertIsInstance(out["dip"], float)
            self.assertIsInstance(out["multimodal"], bool)

    def test_nan_values_filtered(self) -> None:
        vals = np.array([0.38, 0.40, float("nan"), 0.42, 0.44])
        out = rq66.hartigans_dip(vals)
        # Should not raise; n=4 (after nan filter) >= 3 so non-trivial.
        self.assertIn(out["method"], ("scipy_unavailable",
                                      "scipy_stats_dip_test", "trivial_lt3"))


# ----------------------------------------------------------- _summarise_lambda
class TestSummariseLambda(unittest.TestCase):
    def test_unimodal_threshold_distribution(self) -> None:
        thr = np.full(100, 0.38)
        oob = np.full(100, 1.04)
        s = rq66._summarise_lambda(thr, oob)
        td = s["threshold_distribution"]
        od = s["oob_cpwer_distribution"]
        self.assertEqual(td["n_modes_5pct"], 1)
        self.assertAlmostEqual(td["median"], 0.38, places=6)
        self.assertAlmostEqual(td["interval_width"], 0.0, places=6)
        self.assertEqual(td["n_unique"], 1)
        self.assertAlmostEqual(od["median"], 1.04, places=6)
        self.assertEqual(od["n_valid"], 100)

    def test_bimodal_threshold_distribution(self) -> None:
        thr = np.concatenate([np.full(60, 0.38), np.full(40, 0.01)])
        oob = np.full(100, 1.04)
        s = rq66._summarise_lambda(thr, oob)
        td = s["threshold_distribution"]
        self.assertEqual(td["n_modes_5pct"], 2)
        self.assertEqual(td["n_unique"], 2)
        # Two modes: 0.38 (60%), 0.01 (40%).
        mode_thresholds = sorted(m["threshold"] for m in td["modes_5pct"])
        self.assertAlmostEqual(mode_thresholds[0], 0.01, places=6)
        self.assertAlmostEqual(mode_thresholds[1], 0.38, places=6)

    def test_oob_nan_filtered(self) -> None:
        thr = np.full(100, 0.38)
        oob = np.full(100, 1.04)
        oob[0:5] = float("nan")  # 5 NaNs -> 95 valid
        s = rq66._summarise_lambda(thr, oob)
        self.assertEqual(s["oob_cpwer_distribution"]["n_valid"], 95)

    def test_hartigans_dip_included_in_summary(self) -> None:
        thr = np.full(50, 0.38)
        oob = np.full(50, 1.04)
        s = rq66._summarise_lambda(thr, oob)
        self.assertIn("hartigans_dip", s["threshold_distribution"])
        self.assertIn("method", s["threshold_distribution"]["hartigans_dip"])

    def test_oob_frac_below_rq44_median(self) -> None:
        # All OOB = 1.04 < 1.056 -> frac_below_rq44_median = 1.0.
        thr = np.full(50, 0.38)
        oob = np.full(50, 1.04)
        s = rq66._summarise_lambda(thr, oob)
        self.assertAlmostEqual(
            s["oob_cpwer_distribution"]["frac_below_rq44_median"], 1.0, places=6)


# ----------------------------------------------------------- select_best_lambda
class TestSelectBestLambda(unittest.TestCase):
    def _build_summary(self, items):
        return {
            str(item["lambda"]): item for item in items
        }

    def test_deployable_fewest_modes_wins(self) -> None:
        items = [
            {"lambda": 0.0, "n_modes_5pct": 3, "threshold_interval_width": 0.94,
             "oob_cpwer_median": 1.05, "oob_cpwer_interval_width": 0.20},
            {"lambda": 0.5, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.04, "oob_cpwer_interval_width": 0.10},
            {"lambda": 1.0, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.04, "oob_cpwer_interval_width": 0.10},
        ]
        best = rq66.select_best_lambda(self._build_summary(items))
        self.assertAlmostEqual(best["lambda"], 0.5, places=6)
        self.assertTrue(best["all_hypotheses_supported"])

    def test_tie_break_smallest_lambda(self) -> None:
        # Two lambdas tie on modes (1) and width (0.0) and OOB (1.04) ->
        # smallest lambda wins.
        items = [
            {"lambda": 0.5, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.04, "oob_cpwer_interval_width": 0.10},
            {"lambda": 1.0, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.04, "oob_cpwer_interval_width": 0.10},
        ]
        best = rq66.select_best_lambda(self._build_summary(items))
        self.assertAlmostEqual(best["lambda"], 0.5, places=6)

    def test_non_deployable_falls_back_to_closest_oob(self) -> None:
        # All OOB >= 1.056 -> fallback: closest to 1.056, then fewest modes.
        items = [
            {"lambda": 0.0, "n_modes_5pct": 3, "threshold_interval_width": 0.94,
             "oob_cpwer_median": 1.10, "oob_cpwer_interval_width": 0.20},
            {"lambda": 0.5, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.07, "oob_cpwer_interval_width": 0.10},
            {"lambda": 1.0, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.08, "oob_cpwer_interval_width": 0.10},
        ]
        best = rq66.select_best_lambda(self._build_summary(items))
        # Closest to 1.056: 1.07 (diff 0.014) < 1.08 (diff 0.024) < 1.10 (0.044).
        self.assertAlmostEqual(best["lambda"], 0.5, places=6)
        self.assertFalse(best["all_hypotheses_supported"])

    def test_returns_all_expected_fields(self) -> None:
        items = [
            {"lambda": 0.5, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.04, "oob_cpwer_interval_width": 0.10},
        ]
        best = rq66.select_best_lambda(self._build_summary(items))
        for key in ("lambda", "lambda_key", "reason", "all_hypotheses_supported",
                    "n_modes_5pct", "threshold_interval_width",
                    "oob_cpwer_median", "oob_cpwer_interval_width"):
            self.assertIn(key, best)

    def test_all_hypotheses_supported_false_when_h66b_killed(self) -> None:
        items = [
            {"lambda": 0.5, "n_modes_5pct": 1, "threshold_interval_width": 0.0,
             "oob_cpwer_median": 1.06,  # >= 1.056 -> H66b killed
             "oob_cpwer_interval_width": 0.10},
        ]
        best = rq66.select_best_lambda(self._build_summary(items))
        self.assertFalse(best["all_hypotheses_supported"])


# ----------------------------------------------------------- hypothesis helpers
class TestHypothesisHelpers(unittest.TestCase):
    def test_h66a_supported_at_2_modes(self) -> None:
        self.assertTrue(rq66._h66a_supported(2))

    def test_h66a_supported_at_1_mode(self) -> None:
        self.assertTrue(rq66._h66a_supported(1))

    def test_h66a_killed_at_3_modes(self) -> None:
        self.assertFalse(rq66._h66a_supported(3))

    def test_h66b_supported_strict_below(self) -> None:
        self.assertTrue(rq66._h66b_supported(1.055))

    def test_h66b_killed_at_boundary(self) -> None:
        # Strict < 1.056: 1.056 is killed.
        self.assertFalse(rq66._h66b_supported(1.056))

    def test_h66c_supported_strict_below(self) -> None:
        self.assertTrue(rq66._h66c_supported(0.248))

    def test_h66c_killed_at_boundary(self) -> None:
        # Strict < 0.2489: 0.2489 is killed.
        self.assertFalse(rq66._h66c_supported(0.2489))


# ----------------------------------------------------------- verbatim reuse
class TestVerbatimReuse(unittest.TestCase):
    def test_max_across_speakers_is_rq44(self) -> None:
        self.assertIs(rq66.max_across_speakers, rq44.max_across_speakers)

    def test_bootstrap_indices_is_rq44(self) -> None:
        self.assertIs(rq66.bootstrap_indices, rq44.bootstrap_indices)

    def test_out_of_bag_cpwer_is_rq44(self) -> None:
        self.assertIs(rq66.out_of_bag_cpwer, rq44.out_of_bag_cpwer)

    def test_percentile_interval_is_rq44(self) -> None:
        self.assertIs(rq66.percentile_interval, rq44.percentile_interval)

    def test_calibrate_f1_is_rq48(self) -> None:
        self.assertIs(rq66.calibrate_f1, rq48.calibrate_f1)

    def test_count_modes_is_rq48(self) -> None:
        self.assertIs(rq66.count_modes, rq48.count_modes)

    def test_bootstrap_indices_deterministic(self) -> None:
        # Same seed -> same draw (RQ44's contract).
        idx1 = rq66.bootstrap_indices(77, 10, 42)
        idx2 = rq66.bootstrap_indices(77, 10, 42)
        np.testing.assert_array_equal(idx1, idx2)

    def test_bootstrap_indices_shape(self) -> None:
        idx = rq66.bootstrap_indices(77, 50, 42)
        self.assertEqual(idx.shape, (50, 77))

    def test_count_modes_unimodal(self) -> None:
        thr = np.full(100, 0.38)
        modes = rq66.count_modes(thr, 0.05)
        self.assertEqual(modes["n_modes"], 1)

    def test_count_modes_bimodal(self) -> None:
        thr = np.concatenate([np.full(60, 0.38), np.full(40, 0.01)])
        modes = rq66.count_modes(thr, 0.05)
        self.assertEqual(modes["n_modes"], 2)


# ----------------------------------------------------------- real-data smoke tests
class TestRealDataSmoke(unittest.TestCase):
    def test_in_sample_calibration_at_038_for_all_lambdas(self) -> None:
        # On the full 77 windows, 0.38 is both F1-optimal AND the prior mean,
        # so shrinkage has no in-sample effect -> threshold = 0.38 for all lambdas.
        lang_ent, _, _, labels = _load_real_signals()
        for lam in rq66.LAMBDAS:
            out = rq66.calibrate_shrinkage_f1(lang_ent, labels, lam=lam)
            self.assertAlmostEqual(out["threshold"], 0.38, places=6,
                                   msg=f"threshold != 0.38 at lam={lam}")

    def test_in_sample_f1_matches_rq48(self) -> None:
        lang_ent, _, _, labels = _load_real_signals()
        out_combined = rq66.calibrate_shrinkage_f1(lang_ent, labels, lam=0.0)
        out_rq48 = rq48.calibrate_f1(lang_ent, labels)
        self.assertAlmostEqual(out_combined["f1"], out_rq48["f1"], places=6)

    def test_in_sample_f1_is_0933(self) -> None:
        lang_ent, _, _, labels = _load_real_signals()
        out = rq66.calibrate_shrinkage_f1(lang_ent, labels, lam=0.5)
        self.assertAlmostEqual(out["f1"], 0.933333, places=5)

    def test_in_sample_expected_cpwer_is_1043(self) -> None:
        # RQ44's in-sample corrected-router cpWER at threshold 0.38 = 1.043.
        lang_ent, mixed, sep, labels = _load_real_signals()
        out = rq66.calibrate_shrinkage_f1(lang_ent, labels, lam=0.5)
        flag = lang_ent >= out["threshold"] - rq66.EPS
        selected = np.where(flag, mixed, sep)
        self.assertAlmostEqual(float(selected.mean()), 1.04329, places=4)

    def test_n_windows_is_77(self) -> None:
        lang_ent, _, _, _ = _load_real_signals()
        self.assertEqual(len(lang_ent), 77)

    def test_label_counts_37_40(self) -> None:
        _, _, sep, _ = _load_real_signals()
        labels = (sep > rq66.CATASTROPHIC_CPWER).astype(int)
        self.assertEqual(int(labels.sum()), 37)
        self.assertEqual(int((labels == 0).sum()), 40)

    def test_beta22_in_sample_threshold_near_05(self) -> None:
        # Beta(2,2) shrinks toward 0.5; on real data the in-sample threshold
        # should be near 0.5 (the F1-near-optimal threshold closest to 0.5).
        lang_ent, _, _, labels = _load_real_signals()
        out = rq66.calibrate_shrinkage_f1_beta22(lang_ent, labels)
        self.assertGreaterEqual(out["threshold"], 0.4)
        self.assertLessEqual(out["threshold"], 0.5)


# ----------------------------------------------------------- OOB cpWER smoke
class TestOOBCpWERSmoke(unittest.TestCase):
    def test_oob_cpwer_on_single_resample(self) -> None:
        # On one bootstrap resample, OOB cpWER must be in a sane range.
        lang_ent, mixed, sep, labels = _load_real_signals()
        n = len(lang_ent)
        boot_idx = rq44.bootstrap_indices(n, 1, 42)[0]
        out_cal = rq66.calibrate_shrinkage_f1(
            lang_ent[boot_idx], labels[boot_idx], lam=0.5)
        oob = rq66.out_of_bag_cpwer(
            lang_ent, mixed, sep, out_cal["threshold"], boot_idx)
        self.assertGreater(oob["n_oob"], 0)
        self.assertGreaterEqual(oob["cpwer"], 0.5)
        self.assertLessEqual(oob["cpwer"], 2.0)
        self.assertEqual(oob["n_flagged_mixed"] + oob["n_separated"], oob["n_oob"])

    def test_oob_cpwer_empty_when_all_in_bag(self) -> None:
        # If every index is in-bag, OOB is empty -> cpwer is nan.
        lang_ent, mixed, sep, _ = _load_real_signals()
        n = len(lang_ent)
        in_bag = np.arange(n)  # all in-bag
        oob = rq66.out_of_bag_cpwer(lang_ent, mixed, sep, 0.38, in_bag)
        self.assertEqual(oob["n_oob"], 0)
        self.assertTrue(math.isnan(oob["cpwer"]))

    def test_oob_cpwer_threshold_038_routes_correctly(self) -> None:
        # At threshold 0.38, windows with lang_ent >= 0.38 route MIXED.
        lang_ent, mixed, sep, _ = _load_real_signals()
        n = len(lang_ent)
        # Half in-bag, half OOB.
        in_bag = np.arange(0, n, 2)
        oob = rq66.out_of_bag_cpwer(lang_ent, mixed, sep, 0.38, in_bag)
        self.assertGreater(oob["n_oob"], 0)
        # OOB set is odd-indexed windows (1, 3, 5, ...).
        oob_idx = np.setdiff1d(np.arange(n), np.unique(in_bag))
        expected_flagged = int(np.sum(lang_ent[oob_idx] >= 0.38 - rq66.EPS))
        self.assertEqual(oob["n_flagged_mixed"], expected_flagged)


# ----------------------------------------------------------- end-to-end small bootstrap
class TestEndToEndSmallBootstrap(unittest.TestCase):
    def test_small_bootstrap_paired_across_lambdas(self) -> None:
        # Run a small B=10 bootstrap; verify thresholds and OOB cpWER arrays
        # have the right shape and are finite (or NaN only when OOB is empty).
        lang_ent, mixed, sep, labels = _load_real_signals()
        n = len(lang_ent)
        grid_arr = np.asarray(rq66.THRESHOLD_GRID, dtype=float)
        boot_idx = rq44.bootstrap_indices(n, 10, 42)
        for lam in rq66.LAMBDAS:
            thr_arr = np.empty(10, dtype=float)
            oob_arr = np.empty(10, dtype=float)
            for b in range(10):
                idx = boot_idx[b]
                tp, fp, tn, fn, n_pos, n_neg = rq66._confusion_arrays(
                    lang_ent[idx], labels[idx], grid_arr)
                sens, spec = rq66._sens_spec(tp, fp, tn, fn, n_pos, n_neg)
                f1 = rq66._f1_array(tp, fp, fn, sens)
                cal = rq66._select_shrinkage_f1(
                    grid_arr, f1, sens, spec, tp, fp, tn, fn, 0.38, lam)
                thr_arr[b] = cal["threshold"]
                oob = rq66.out_of_bag_cpwer(
                    lang_ent, mixed, sep, cal["threshold"], idx)
                oob_arr[b] = oob["cpwer"]
            self.assertEqual(thr_arr.shape, (10,))
            self.assertEqual(oob_arr.shape, (10,))
            # All thresholds are in grid.
            for t in thr_arr:
                self.assertIn(float(t), [float(g) for g in grid_arr])

    def test_shrinkage_monotone_modes_on_real_data_small_b(self) -> None:
        # On a small B=50 bootstrap, modes should be non-increasing in lambda
        # (shrinkage can only collapse modes, not create them). This is the
        # core mechanism: shrinkage pulls toward 0.38, collapsing multimodality.
        lang_ent, mixed, sep, labels = _load_real_signals()
        n = len(lang_ent)
        grid_arr = np.asarray(rq66.THRESHOLD_GRID, dtype=float)
        boot_idx = rq44.bootstrap_indices(n, 50, 42)
        prev_modes = None
        for lam in rq66.LAMBDAS:
            thr_arr = np.empty(50, dtype=float)
            for b in range(50):
                idx = boot_idx[b]
                tp, fp, tn, fn, n_pos, n_neg = rq66._confusion_arrays(
                    lang_ent[idx], labels[idx], grid_arr)
                sens, spec = rq66._sens_spec(tp, fp, tn, fn, n_pos, n_neg)
                f1 = rq66._f1_array(tp, fp, fn, sens)
                cal = rq66._select_shrinkage_f1(
                    grid_arr, f1, sens, spec, tp, fp, tn, fn, 0.38, lam)
                thr_arr[b] = cal["threshold"]
            modes = rq66.count_modes(thr_arr, 0.05)
            if prev_modes is not None:
                self.assertLessEqual(modes["n_modes"], prev_modes,
                                     msg=f"modes increased at lam={lam}")
            prev_modes = modes["n_modes"]


if __name__ == "__main__":
    unittest.main()
