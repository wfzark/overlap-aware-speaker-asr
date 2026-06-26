"""Tests for RQ49: speaker-count stratified threshold (experimental/frontier).

Pins the pure helpers: ``stratify_by_speaker_count``,
``calibrate_threshold_at_spec`` (RQ44 verbatim), ``bootstrap_stratified``,
``combined_oob_cpwer``. Also pins the active-speaker counter (RQ38 verbatim)
and two smoke tests on the real 77-window AISHELL-4 data:

  - pooled in-sample calibration reproduces RQ44/RQ25's 0.38 threshold
  - the <= 2 active-speaker stratum includes BOTH Mode S windows (w22, w30)

No Whisper / no audio needed. numpy + stdlib only.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# The RQ49 analysis script lives in results/frontier/ as a standalone module
# (no src. package). Import it via sys.path manipulation, mirroring the
# harness test_bootstrap_threshold / test_rq40_mode_s_corpus_specificity pattern.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _PROJECT_ROOT / "results" / "frontier" / "stratified_threshold"
sys.path.insert(0, str(_SCRIPT_DIR))

import stratified_threshold_analysis as rq49  # noqa: E402  (path-injected import)

AISHELL4_JSON = (
    _PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)
RESULTS_JSON = (
    _SCRIPT_DIR / "stratified_threshold_results.json"
)
RESULTS_CSV = (
    _SCRIPT_DIR / "stratified_threshold_results.csv"
)


# --------------------------------------------- stratify_by_speaker_count
class TestStratifyBySpeakerCount(unittest.TestCase):
    def test_returns_two_arrays(self) -> None:
        le, gt = rq49.stratify_by_speaker_count(np.array([1, 2, 3, 4]))
        self.assertEqual(len(le) + len(gt), 4)

    def test_partition_is_exhaustive_and_disjoint(self) -> None:
        active = np.array([0, 1, 2, 3, 4, 5])
        le, gt = rq49.stratify_by_speaker_count(active)
        union = sorted(list(le) + list(gt))
        self.assertEqual(union, [0, 1, 2, 3, 4, 5])
        self.assertEqual(set(le) & set(gt), set())

    def test_stratum1_is_le_split(self) -> None:
        active = np.array([0, 1, 2, 3, 4])
        le, _ = rq49.stratify_by_speaker_count(active, split=2)
        for i in le:
            self.assertLessEqual(active[i], 2)

    def test_stratum2_is_gt_split(self) -> None:
        active = np.array([0, 1, 2, 3, 4])
        _, gt = rq49.stratify_by_speaker_count(active, split=2)
        for i in gt:
            self.assertGreater(active[i], 2)

    def test_default_split_is_2(self) -> None:
        active = np.array([1, 2, 3])
        le_default, gt_default = rq49.stratify_by_speaker_count(active)
        le_explicit, gt_explicit = rq49.stratify_by_speaker_count(active, split=2)
        np.testing.assert_array_equal(le_default, le_explicit)
        np.testing.assert_array_equal(gt_default, gt_explicit)

    def test_custom_split(self) -> None:
        active = np.array([1, 2, 3, 4, 5])
        le, gt = rq49.stratify_by_speaker_count(active, split=3)
        self.assertEqual(sorted(le), [0, 1, 2])   # active 1,2,3
        self.assertEqual(sorted(gt), [3, 4])       # active 4,5

    def test_empty_input(self) -> None:
        le, gt = rq49.stratify_by_speaker_count(np.array([], dtype=int))
        self.assertEqual(len(le), 0)
        self.assertEqual(len(gt), 0)

    def test_all_le_split_goes_to_stratum1(self) -> None:
        active = np.array([0, 1, 2])
        le, gt = rq49.stratify_by_speaker_count(active, split=2)
        self.assertEqual(len(le), 3)
        self.assertEqual(len(gt), 0)

    def test_all_gt_split_goes_to_stratum2(self) -> None:
        active = np.array([3, 4, 5])
        le, gt = rq49.stratify_by_speaker_count(active, split=2)
        self.assertEqual(len(le), 0)
        self.assertEqual(len(gt), 3)

    def test_boundary_equal_goes_to_stratum1(self) -> None:
        # active == split goes to stratum 1 (<=).
        active = np.array([2, 2])
        le, gt = rq49.stratify_by_speaker_count(active, split=2)
        self.assertEqual(len(le), 2)
        self.assertEqual(len(gt), 0)


# --------------------------------------------- calibrate_threshold_at_spec
class TestCalibrateThresholdAtSpec(unittest.TestCase):
    def test_separable_case_high_sensitivity(self) -> None:
        # negs in [0, 0.20], pos at 1.0/1.5. A threshold just above 0.20
        # separates them: sens=1, spec=1.
        scores = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 1.0, 1.5])
        labels = np.array([0, 0, 0, 0, 0, 1, 1])
        out = rq49.calibrate_threshold_at_spec(scores, labels)
        self.assertGreaterEqual(out["specificity"], 0.90 - 1e-9)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["tp"], 2)
        self.assertEqual(out["fn"], 0)

    def test_target_specificity_respected(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1])
        out = rq49.calibrate_threshold_at_spec(scores, labels, target_spec=0.75)
        self.assertGreaterEqual(out["specificity"], 0.75 - 1e-9)

    def test_all_negative_no_positives(self) -> None:
        # No positives -> sensitivity is 0 for every threshold; specificity is
        # 1.0 everywhere. The tie-breaker picks the lowest threshold.
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        out = rq49.calibrate_threshold_at_spec(scores, labels)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertEqual(out["specificity"], 1.0)
        self.assertEqual(out["tp"], 0)
        self.assertEqual(out["fn"], 0)

    def test_all_positive_no_negatives(self) -> None:
        # No negatives -> specificity is 1.0 (n_neg=0 branch); sensitivity is
        # maximised by flagging everything -> lowest threshold.
        scores = np.array([0.5, 0.6, 0.9])
        labels = np.array([1, 1, 1])
        out = rq49.calibrate_threshold_at_spec(scores, labels)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["specificity"], 1.0)
        self.assertEqual(out["threshold"], 0.0)

    def test_tie_break_lower_threshold(self) -> None:
        # negs at 0,1 ; pos at 5,6. Several thresholds give sens=1 & spec=1;
        # the tie-breaker selects the lowest such threshold.
        scores = np.array([0.0, 1.0, 5.0, 6.0])
        labels = np.array([0, 0, 1, 1])
        out = rq49.calibrate_threshold_at_spec(scores, labels)
        self.assertEqual(out["sensitivity"], 1.0)
        self.assertEqual(out["specificity"], 1.0)
        # Lowest threshold that keeps both negs below (>= threshold flags pos).
        # negs 0,1 must be below threshold -> threshold > 1.0; grid step 0.01
        # so the lowest such threshold is 1.01.
        self.assertGreater(out["threshold"], 1.0)
        self.assertLessEqual(out["threshold"], 1.01 + 1e-9)

    def test_empty_input_returns_lowest_threshold(self) -> None:
        # With no data, n_neg=0 -> specificity is trivially 1.0 for every
        # threshold (the n_neg==0 branch), so the target is always met and the
        # tie-breaker (lowest threshold) wins. Verbatim RQ44 behaviour.
        out = rq49.calibrate_threshold_at_spec(np.array([]), np.array([]))
        self.assertEqual(out["threshold"], 0.0)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertEqual(out["specificity"], 1.0)

    def test_no_threshold_meets_spec_falls_back_to_grid_max(self) -> None:
        # All negatives at 2.0 + one positive at 2.0: even the highest grid
        # threshold (2.0) flags every negative (fp=10, tn=0 -> spec=0.0 < 0.9),
        # so NO threshold satisfies the specificity target and the function
        # falls back to grid[-1] = 2.0 (most conservative: flag nothing).
        scores = np.array([2.0] * 10 + [2.0])
        labels = np.array([0] * 10 + [1])
        out = rq49.calibrate_threshold_at_spec(scores, labels)
        self.assertEqual(out["threshold"], 2.0)
        self.assertEqual(out["sensitivity"], 0.0)
        self.assertEqual(out["specificity"], 1.0)

    def test_custom_grid(self) -> None:
        scores = np.array([0.0, 0.5, 1.0])
        labels = np.array([0, 0, 1])
        out = rq49.calibrate_threshold_at_spec(scores, labels, grid=[0.4, 0.6, 0.9])
        self.assertIn(out["threshold"], [0.4, 0.6, 0.9])

    def test_returns_dict_with_required_keys(self) -> None:
        out = rq49.calibrate_threshold_at_spec(np.array([0.0, 1.0]), np.array([0, 1]))
        for k in ("threshold", "sensitivity", "specificity", "tp", "fp", "tn", "fn"):
            self.assertIn(k, out)


# --------------------------------------------- combined_oob_cpwer
class TestCombinedOobCpwer(unittest.TestCase):
    def test_all_separated_below_threshold(self) -> None:
        # All OOB scores below both thresholds -> everything routed SEPARATED.
        s1 = np.array([0.1, 0.2]); m1 = np.array([2.0, 3.0]); p1 = np.array([1.0, 1.0])
        s2 = np.array([0.15]); m2 = np.array([2.0]); p2 = np.array([1.0])
        out = rq49.combined_oob_cpwer(s1, m1, p1, 0.5, s2, m2, p2, 0.5)
        self.assertEqual(out["cpwer"], 1.0)
        self.assertEqual(out["n_oob"], 3)
        self.assertEqual(out["n_flagged_mixed"], 0)
        self.assertEqual(out["n_separated"], 3)

    def test_all_mixed_above_threshold(self) -> None:
        s1 = np.array([0.9, 0.8]); m1 = np.array([1.0, 1.0]); p1 = np.array([2.0, 3.0])
        s2 = np.array([0.95]); m2 = np.array([1.0]); p2 = np.array([4.0])
        out = rq49.combined_oob_cpwer(s1, m1, p1, 0.5, s2, m2, p2, 0.5)
        self.assertEqual(out["cpwer"], 1.0)
        self.assertEqual(out["n_oob"], 3)
        self.assertEqual(out["n_flagged_mixed"], 3)
        self.assertEqual(out["n_separated"], 0)

    def test_stratum_specific_thresholds(self) -> None:
        # Stratum 1 threshold 0.5 (score 0.6 flagged mixed), stratum 2
        # threshold 0.95 (score 0.6 NOT flagged -> separated). Same score,
        # different routing per stratum.
        s1 = np.array([0.6]); m1 = np.array([1.0]); p1 = np.array([2.0])
        s2 = np.array([0.6]); m2 = np.array([1.0]); p2 = np.array([2.0])
        out = rq49.combined_oob_cpwer(s1, m1, p1, 0.5, s2, m2, p2, 0.95)
        # stratum 1 -> mixed (1.0); stratum 2 -> separated (2.0); mean = 1.5
        self.assertAlmostEqual(out["cpwer"], 1.5, places=6)
        self.assertEqual(out["n_flagged_mixed"], 1)
        self.assertEqual(out["n_separated"], 1)
        self.assertEqual(out["n_oob_s1"], 1)
        self.assertEqual(out["n_oob_s2"], 1)

    def test_empty_both_returns_nan(self) -> None:
        out = rq49.combined_oob_cpwer(
            np.array([]), np.array([]), np.array([]), 0.5,
            np.array([]), np.array([]), np.array([]), 0.5,
        )
        self.assertTrue(math.isnan(out["cpwer"]))
        self.assertEqual(out["n_oob"], 0)

    def test_one_stratum_empty(self) -> None:
        s1 = np.array([0.6]); m1 = np.array([1.0]); p1 = np.array([2.0])
        out = rq49.combined_oob_cpwer(
            s1, m1, p1, 0.5,
            np.array([]), np.array([]), np.array([]), 0.5,
        )
        # Only stratum 1 contributes: score 0.6 >= 0.5 -> mixed -> cpwer 1.0.
        self.assertEqual(out["cpwer"], 1.0)
        self.assertEqual(out["n_oob"], 1)
        self.assertEqual(out["n_oob_s2"], 0)

    def test_counts_and_mean_correct(self) -> None:
        # 2 stratum-1 windows: one mixed (1.0), one separated (2.0).
        # 1 stratum-2 window: separated (1.5). mean = (1.0+2.0+1.5)/3 = 1.5.
        s1 = np.array([0.9, 0.1]); m1 = np.array([1.0, 2.0]); p1 = np.array([2.0, 2.0])
        s2 = np.array([0.1]); m2 = np.array([9.0]); p2 = np.array([1.5])
        out = rq49.combined_oob_cpwer(s1, m1, p1, 0.5, s2, m2, p2, 0.5)
        self.assertAlmostEqual(out["cpwer"], (1.0 + 2.0 + 1.5) / 3.0, places=6)
        self.assertEqual(out["n_flagged_mixed"], 1)
        self.assertEqual(out["n_separated"], 2)


# --------------------------------------------- bootstrap_stratified
class TestBootstrapStratified(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 8 synthetic windows: stratum1 = first 4 (<=2), stratum2 = last 4 (>2).
        cls.scores = np.array([0.1, 0.2, 0.9, 1.5, 0.3, 0.4, 1.2, 1.8])
        cls.labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        cls.mixed = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        cls.sep = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
        cls.s1 = np.array([0, 1, 2, 3])
        cls.s2 = np.array([4, 5, 6, 7])
        cls.out = rq49.bootstrap_stratified(
            cls.scores, cls.labels, cls.mixed, cls.sep,
            cls.s1, cls.s2, n_boot=10, seed=42,
        )

    def test_returns_correct_shapes(self) -> None:
        self.assertEqual(self.out["thresholds_1"].shape, (10,))
        self.assertEqual(self.out["thresholds_2"].shape, (10,))
        self.assertEqual(self.out["oob_cpwer_combined"].shape, (10,))
        self.assertEqual(self.out["n_oob_1"].shape, (10,))

    def test_deterministic_with_seed(self) -> None:
        out2 = rq49.bootstrap_stratified(
            self.scores, self.labels, self.mixed, self.sep,
            self.s1, self.s2, n_boot=10, seed=42,
        )
        np.testing.assert_array_equal(self.out["thresholds_1"], out2["thresholds_1"])
        np.testing.assert_array_equal(self.out["thresholds_2"], out2["thresholds_2"])
        np.testing.assert_array_equal(self.out["oob_cpwer_combined"], out2["oob_cpwer_combined"])

    def test_thresholds_in_grid_range(self) -> None:
        for t in self.out["thresholds_1"]:
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 2.0)
        for t in self.out["thresholds_2"]:
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 2.0)

    def test_combined_oob_nonneg_or_nan(self) -> None:
        for c in self.out["oob_cpwer_combined"]:
            self.assertTrue(math.isnan(c) or c >= 0.0)

    def test_n1_n2_in_summary(self) -> None:
        self.assertEqual(self.out["n1"], 4)
        self.assertEqual(self.out["n2"], 4)
        self.assertEqual(self.out["n_boot"], 10)
        self.assertEqual(self.out["seed"], 42)

    def test_different_seed_differs(self) -> None:
        out2 = rq49.bootstrap_stratified(
            self.scores, self.labels, self.mixed, self.sep,
            self.s1, self.s2, n_boot=10, seed=7,
        )
        self.assertFalse(np.array_equal(self.out["thresholds_1"], out2["thresholds_1"]))


# --------------------------------------------- count_modes helper
class TestCountModes(unittest.TestCase):
    def test_single_mode(self) -> None:
        arr = np.array([0.38] * 100)
        md = rq49.count_modes(arr, 0.05)
        self.assertEqual(md["n_modes"], 1)
        self.assertEqual(md["n_unique"], 1)

    def test_six_modes_five_above_threshold(self) -> None:
        # RQ44-like: 5 values >= 5%, 1 value < 5%.
        arr = np.array([0.38] * 60 + [0.87] * 15 + [0.01] * 9 + [0.95] * 8 + [0.33] * 6 + [1.5] * 2)
        md = rq49.count_modes(arr, 0.05)
        self.assertEqual(md["n_modes"], 5)
        self.assertEqual(md["n_unique"], 6)

    def test_empty(self) -> None:
        md = rq49.count_modes(np.array([]), 0.05)
        self.assertEqual(md["n_modes"], 0)
        self.assertEqual(md["n_unique"], 0)

    def test_modes_sorted_by_count(self) -> None:
        arr = np.array([0.5] * 30 + [0.2] * 10)
        md = rq49.count_modes(arr, 0.05)
        self.assertEqual(md["modes"][0]["threshold"], 0.5)
        self.assertEqual(md["modes"][1]["threshold"], 0.2)


# --------------------------------------------- smoke: real AISHELL-4 data
@unittest.skipUnless(AISHELL4_JSON.exists(), "AISHELL-4 validation JSON not present")
class TestInSampleReproductionSmoke(unittest.TestCase):
    """Smoke test: pooled in-sample calibration reproduces RQ44/RQ25's 0.38
    threshold on the real 77-window AISHELL-4 data."""

    @classmethod
    def setUpClass(cls) -> None:
        data = json.loads(AISHELL4_JSON.read_text(encoding="utf-8"))
        windows = data["windows"]
        cls.windows = windows
        cls.lang_ent = np.array(
            [rq49.max_across_speakers(w) for w in windows], dtype=float
        )
        cls.sep_cpwer = np.array(
            [float(w["always_separated_cpwer"]) for w in windows], dtype=float
        )
        cls.active = np.array(
            [rq49.count_active_speakers(w.get("separated_text_per_speaker", {})) for w in windows],
            dtype=int,
        )
        cls.labels = (cls.sep_cpwer > 1.0).astype(int)

    def test_pooled_in_sample_threshold_is_038(self) -> None:
        out = rq49.calibrate_threshold_at_spec(self.lang_ent, self.labels)
        self.assertAlmostEqual(out["threshold"], 0.38, places=6)

    def test_n_windows_is_77_and_labels_match_rq44(self) -> None:
        self.assertEqual(len(self.windows), 77)
        self.assertEqual(int(self.labels.sum()), 37)
        self.assertEqual(int((self.labels == 0).sum()), 40)


# --------------------------------------------- smoke: Mode S in stratum 1
@unittest.skipUnless(AISHELL4_JSON.exists(), "AISHELL-4 validation JSON not present")
class TestModeSInStratum1Smoke(unittest.TestCase):
    """Smoke test: the <= 2 active-speaker stratum includes BOTH Mode S
    windows (w22 and w30, per RQ19/RQ38)."""

    @classmethod
    def setUpClass(cls) -> None:
        data = json.loads(AISHELL4_JSON.read_text(encoding="utf-8"))
        windows = data["windows"]
        cls.window_ids = [int(w["window_id"]) for w in windows]
        active = np.array(
            [rq49.count_active_speakers(w.get("separated_text_per_speaker", {})) for w in windows],
            dtype=int,
        )
        cls.s1_idx, cls.s2_idx = rq49.stratify_by_speaker_count(active, 2)
        cls.s1_ids = {cls.window_ids[i] for i in cls.s1_idx}

    def test_mode_s_window_22_in_stratum1(self) -> None:
        self.assertIn(22, self.s1_ids)

    def test_mode_s_window_30_in_stratum1(self) -> None:
        self.assertIn(30, self.s1_ids)

    def test_stratum_sizes_62_and_15(self) -> None:
        # Active-speaker definition (RQ38): <=2 -> 62, >2 -> 15.
        self.assertEqual(len(self.s1_idx), 62)
        self.assertEqual(len(self.s2_idx), 15)

    def test_strata_partition_all_77_windows(self) -> None:
        self.assertEqual(len(self.s1_idx) + len(self.s2_idx), 77)


# --------------------------------------------- integration: result files
@unittest.skipUnless(RESULTS_JSON.exists(), "RQ49 results JSON not yet generated")
class TestResultsFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    def test_label_is_experimental_frontier(self) -> None:
        self.assertEqual(self.summary["label"], "experimental/frontier")

    def test_n_windows_is_77(self) -> None:
        self.assertEqual(self.summary["n_windows"], 77)

    def test_stratum_sizes_recorded(self) -> None:
        self.assertEqual(self.summary["stratification"]["stratum1"]["n_windows"], 62)
        self.assertEqual(self.summary["stratification"]["stratum2"]["n_windows"], 15)

    def test_stratum2_hallucination_rate_93pct(self) -> None:
        rate = self.summary["stratification"]["stratum2"]["hallucination_rate"]
        self.assertAlmostEqual(rate, 14 / 15, places=4)

    def test_mode_s_in_stratum1_recorded(self) -> None:
        self.assertEqual(
            self.summary["stratification"]["mode_s_in_stratum1"], [22, 30]
        )

    def test_hypothesis_verdicts_present(self) -> None:
        verdicts = self.summary["hypothesis_verdicts"]
        for h in ("H49a", "H49b", "H49c"):
            self.assertIn(h, verdicts)
            self.assertIn("supported", verdicts[h])
            self.assertIn("statement", verdicts[h])

    def test_h49a_killed_with_4_modes(self) -> None:
        # Pre-registered: kill if > 3 modes (>= 5%). Observed: 4 modes.
        v = self.summary["hypothesis_verdicts"]["H49a"]
        self.assertEqual(v["n_modes_5pct"], 4)
        self.assertFalse(v["supported"])

    def test_h49b_supported_with_2_modes(self) -> None:
        v = self.summary["hypothesis_verdicts"]["H49b"]
        self.assertEqual(v["n_modes_5pct"], 2)
        self.assertTrue(v["supported"])

    def test_h49c_verdict_and_tie_note(self) -> None:
        v = self.summary["hypothesis_verdicts"]["H49c"]
        # Combined median 1.055556 < 1.056 -> technically supported (rounding).
        self.assertAlmostEqual(v["median_oob_cpwer"], 1.055556, places=5)
        self.assertIn("substantive_note", v)

    def test_pooled_matched_reproduces_rq44_modality(self) -> None:
        # Within-script pooled bootstrap at matched B=2000 must reproduce
        # RQ44's 6 unique / 5 modes (>= 5%) and width 0.94.
        pd = self.summary["pooled_bootstrap_matched"]["threshold_distribution"]
        self.assertEqual(pd["n_unique"], 6)
        self.assertEqual(pd["n_modes_5pct"], 5)
        self.assertAlmostEqual(pd["interval_width"], 0.94, places=4)


if __name__ == "__main__":
    unittest.main()
