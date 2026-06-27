"""Tests for RQ60: KL+lang-id ensemble corrected router.

Pin the PURE helpers used by
``results/frontier/ensemble_corrected_router/ensemble_router_analysis.py``:

  * KL detector primitives imported from ``src.llm_semantic_critic`` (RQ34/RQ58):
    ``char_ngrams``, ``char_distribution``, ``average_distributions``,
    ``kl_divergence``, ``compute_anomaly_score``, ``build_reference_distribution``,
    ``calibrate_threshold_at_specificity``, ``evaluate_at_threshold``,
    ``subgroup_sensitivity``, ``label_window``, ``separated_concat``.
  * lang-id primitives: ``language_id_entropy``, ``max_across_speakers``.
  * RQ58 helpers reused: ``build_kl_reference``, ``compute_kl_scores``,
    ``kl_route_decision``, ``lang_id_route_decision``, ``cpwer_for``.
  * NEW RQ60 ensemble helpers: ``kl_flag``, ``lang_id_flag``, ``or_flag``,
    ``and_flag``, ``or_route_decision``, ``and_route_decision``.
  * Bootstrap + BCa CI helpers (RQ39 framework): ``bootstrap_indices``,
    ``bootstrap_distribution``, ``percentile_ci``, ``_jackknife_means``,
    ``bca_ci``, ``paired_delta_distribution``, ``paired_delta_ci``.

Synthetic data only — no AISHELL-4 file, no MeetEval, no Whisper, no audio, no
LLM. Label: experimental/frontier.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis module importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = (
    PROJECT_ROOT
    / "results"
    / "frontier"
    / "ensemble_corrected_router"
)
sys.path.insert(0, str(SCRIPT_DIR))

from ensemble_router_analysis import (  # noqa: E402
    ALPHA,
    ALWAYS_MIXED_CPWER,
    AND_FLAG_FP_KILL_RATE,
    EPS,
    KL_THRESHOLD,
    LANG_ID_THRESHOLD,
    MODE_S_WINDOW_IDS,
    N_BOOT,
    N_GRAM,
    ORACLE_BEST_CPWER,
    RQ16_LANG_ID_CPWER,
    RQ58_KL_CPWER,
    SEED,
    _jackknife_means,
    and_flag,
    and_route_decision,
    average_distributions,
    bca_ci,
    bootstrap_distribution,
    bootstrap_indices,
    build_kl_reference,
    build_reference_distribution,
    calibrate_threshold_at_specificity,
    char_distribution,
    char_ngrams,
    compute_anomaly_score,
    compute_kl_scores,
    cpwer_for,
    evaluate_at_threshold,
    kl_divergence,
    kl_flag,
    kl_route_decision,
    lang_id_flag,
    lang_id_route_decision,
    language_id_entropy,
    max_across_speakers,
    or_flag,
    or_route_decision,
    paired_delta_ci,
    paired_delta_distribution,
    percentile_ci,
    subgroup_sensitivity,
)


# ======================================================================
# Constants and configuration
# ======================================================================
class TestConstants(unittest.TestCase):
    def test_ngram_is_2(self) -> None:
        self.assertEqual(N_GRAM, 2)

    def test_kl_threshold_matches_rq58(self) -> None:
        # RQ58's empirically-calibrated threshold at >=90% specificity.
        self.assertAlmostEqual(KL_THRESHOLD, 5.418144, places=4)

    def test_lang_id_threshold_is_0_38(self) -> None:
        # Task spec pre-registered threshold 0.38 (identical results to 0.409).
        self.assertEqual(LANG_ID_THRESHOLD, 0.38)

    def test_mode_s_window_ids(self) -> None:
        self.assertEqual(MODE_S_WINDOW_IDS, {22, 30})

    def test_rq58_kl_cpwer_reference(self) -> None:
        self.assertAlmostEqual(RQ58_KL_CPWER, 1.030303, places=5)

    def test_rq16_lang_id_cpwer_reference(self) -> None:
        self.assertAlmostEqual(RQ16_LANG_ID_CPWER, 1.04329, places=4)

    def test_oracle_cpwer_reference(self) -> None:
        self.assertAlmostEqual(ORACLE_BEST_CPWER, 1.017316, places=5)

    def test_always_mixed_cpwer_reference(self) -> None:
        self.assertAlmostEqual(ALWAYS_MIXED_CPWER, 1.17316, places=4)

    def test_and_fp_kill_rate(self) -> None:
        # H60c: AND FP rate killed if >= 7.5% (3/40).
        self.assertAlmostEqual(AND_FLAG_FP_KILL_RATE, 0.075, places=3)

    def test_bootstrap_config(self) -> None:
        self.assertEqual(N_BOOT, 10000)
        self.assertEqual(SEED, 42)
        self.assertAlmostEqual(ALPHA, 0.05, places=2)


# ======================================================================
# char_ngrams (RQ34 primitive)
# ======================================================================
class TestCharNgrams(unittest.TestCase):
    def test_bigram_count_basic(self) -> None:
        grams = char_ngrams("abcd", n=2)
        self.assertEqual(grams, {"ab": 1, "bc": 1, "cd": 1})

    def test_whitespace_stripped(self) -> None:
        grams = char_ngrams("a b c", n=2)
        self.assertEqual(grams, {"ab": 1, "bc": 1})

    def test_repeated_bigram_counted(self) -> None:
        grams = char_ngrams("abab", n=2)
        self.assertEqual(grams, {"ab": 2, "ba": 1})

    def test_short_text_returns_whole_string(self) -> None:
        grams = char_ngrams("a", n=2)
        self.assertEqual(grams, {"a": 1})

    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(char_ngrams("", n=2), {})


# ======================================================================
# char_distribution (RQ34 primitive)
# ======================================================================
class TestCharDistribution(unittest.TestCase):
    def test_distribution_sums_to_one(self) -> None:
        dist = char_distribution("aabbcc", n=2)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)

    def test_uniform_distribution_for_repeated_bigrams(self) -> None:
        dist = char_distribution("abab", n=2)
        self.assertAlmostEqual(dist["ab"], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(dist["ba"], 1.0 / 3.0, places=9)

    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(char_distribution("", n=2), {})

    def test_vocab_restricted_returns_zeros_for_unseen(self) -> None:
        dist = char_distribution("ab", n=2, vocab={"ab", "cd"})
        self.assertAlmostEqual(dist["ab"], 1.0, places=9)
        self.assertAlmostEqual(dist["cd"], 0.0, places=9)


# ======================================================================
# average_distributions (RQ34 primitive)
# ======================================================================
class TestAverageDistributions(unittest.TestCase):
    def test_average_of_two_distributions(self) -> None:
        d1 = {"a": 1.0, "b": 0.0}
        d2 = {"a": 0.0, "b": 1.0}
        avg = average_distributions([d1, d2])
        self.assertAlmostEqual(avg["a"], 0.5, places=9)
        self.assertAlmostEqual(avg["b"], 0.5, places=9)

    def test_average_of_empty_list_returns_empty(self) -> None:
        self.assertEqual(average_distributions([]), {})

    def test_average_keys_are_union(self) -> None:
        d1 = {"a": 1.0}
        d2 = {"b": 1.0}
        avg = average_distributions([d1, d2])
        self.assertEqual(set(avg.keys()), {"a", "b"})


# ======================================================================
# kl_divergence (RQ34 primitive)
# ======================================================================
class TestKlDivergence(unittest.TestCase):
    def test_identical_distributions_kl_zero(self) -> None:
        d = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(kl_divergence(d, d), 0.0, places=6)

    def test_disjoint_distributions_high_kl(self) -> None:
        p = {"a": 1.0}
        q = {"b": 1.0}
        kl = kl_divergence(p, q)
        self.assertGreater(kl, 5.0)

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(kl_divergence({}, {}), 0.0)

    def test_kl_non_negative(self) -> None:
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.4, "b": 0.6}
        self.assertGreaterEqual(kl_divergence(p, q), -EPS)


# ======================================================================
# compute_anomaly_score (RQ34 primitive)
# ======================================================================
class TestComputeAnomalyScore(unittest.TestCase):
    def test_empty_text_returns_zero(self) -> None:
        ref = {"ab": 0.5, "cd": 0.5}
        self.assertEqual(compute_anomaly_score("", ref, n=2), 0.0)

    def test_empty_ref_returns_zero(self) -> None:
        self.assertEqual(compute_anomaly_score("abcd", {}, n=2), 0.0)

    def test_matching_text_low_score(self) -> None:
        ref = char_distribution("abcdab", n=2)
        score = compute_anomaly_score("abcdab", ref, n=2)
        self.assertLess(score, 0.1)


# ======================================================================
# build_reference_distribution (RQ34 primitive)
# ======================================================================
class TestBuildReferenceDistribution(unittest.TestCase):
    def test_build_from_multiple_texts(self) -> None:
        texts = ["abcd", "abef"]
        ref = build_reference_distribution(texts, n=2)
        self.assertIn("ab", ref)
        self.assertGreater(ref["ab"], 0.0)

    def test_empty_texts_filtered(self) -> None:
        ref = build_reference_distribution(["", "  ", "ab"], n=2)
        self.assertGreater(len(ref), 0)


# ======================================================================
# language_id_entropy (RQ13 primitive)
# ======================================================================
class TestLanguageIdEntropy(unittest.TestCase):
    def test_clean_chinese_low_entropy(self) -> None:
        ent = language_id_entropy("你好世界你好")
        self.assertLess(ent, 0.1)

    def test_mixed_scripts_high_entropy(self) -> None:
        ent = language_id_entropy("你好Helloカタカナ세계")
        self.assertGreater(ent, 1.0)

    def test_empty_text_zero_entropy(self) -> None:
        self.assertEqual(language_id_entropy(""), 0.0)

    def test_whitespace_only_zero_entropy(self) -> None:
        self.assertEqual(language_id_entropy("   "), 0.0)


# ======================================================================
# max_across_speakers (RQ12/RQ13 primitive)
# ======================================================================
class TestMaxAcrossSpeakers(unittest.TestCase):
    def test_max_of_multiple_speakers(self) -> None:
        window = {"separated_text_per_speaker": {"a": "你好", "b": "Hello世界"}}
        ent = max_across_speakers(window, language_id_entropy)
        self.assertGreater(ent, 0.5)

    def test_empty_speakers_returns_zero(self) -> None:
        window = {"separated_text_per_speaker": {}}
        self.assertEqual(max_across_speakers(window, language_id_entropy), 0.0)

    def test_whitespace_speakers_skipped(self) -> None:
        window = {"separated_text_per_speaker": {"a": "  ", "b": "  "}}
        self.assertEqual(max_across_speakers(window, language_id_entropy), 0.0)


# ======================================================================
# build_kl_reference (RQ58 helper, reused)
# ======================================================================
class TestBuildKlReference(unittest.TestCase):
    def test_reference_built_from_non_hallucinated(self) -> None:
        labels = [
            {"separated_text": "你好世界", "hallucinated": False},
            {"separated_text": "大家好", "hallucinated": False},
            {"separated_text": "xyzabc", "hallucinated": True},
        ]
        ref = build_kl_reference(labels, n=2)
        self.assertGreater(len(ref), 0)

    def test_reference_ignores_hallucinated(self) -> None:
        labels_neg = [
            {"separated_text": "你好世界", "hallucinated": False},
        ]
        labels_with_pos = labels_neg + [
            {"separated_text": "zzzzzz", "hallucinated": True},
        ]
        ref_neg = build_kl_reference(labels_neg, n=2)
        ref_both = build_kl_reference(labels_with_pos, n=2)
        self.assertEqual(ref_neg, ref_both)


# ======================================================================
# compute_kl_scores (RQ58 helper, reused)
# ======================================================================
class TestComputeKlScores(unittest.TestCase):
    def test_scores_per_window(self) -> None:
        ref = build_reference_distribution(["你好世界你好"], n=2)
        windows = [
            {"separated_text_per_speaker": {"a": "你好世界你好"}},
            {"separated_text_per_speaker": {"a": "xyzabcqwerty"}},
        ]
        scores = compute_kl_scores(windows, ref, n=2)
        self.assertEqual(len(scores), 2)
        self.assertLess(scores[0], scores[1])

    def test_empty_window_returns_zero(self) -> None:
        ref = build_reference_distribution(["你好"], n=2)
        windows = [{"separated_text_per_speaker": {}}]
        scores = compute_kl_scores(windows, ref, n=2)
        self.assertEqual(scores, [0.0])


# ======================================================================
# kl_route_decision (RQ58 helper, reused)
# ======================================================================
class TestKlRouteDecision(unittest.TestCase):
    def test_above_threshold_routes_mixed(self) -> None:
        self.assertEqual(kl_route_decision(10.0, 5.42), "mixed")

    def test_below_threshold_routes_separated(self) -> None:
        self.assertEqual(kl_route_decision(1.0, 5.42), "separated")

    def test_at_threshold_routes_mixed(self) -> None:
        self.assertEqual(kl_route_decision(5.42, 5.42), "mixed")


# ======================================================================
# lang_id_route_decision (RQ58 helper, reused)
# ======================================================================
class TestLangIdRouteDecision(unittest.TestCase):
    def test_high_entropy_routes_mixed(self) -> None:
        window = {"separated_text_per_speaker": {"a": "Helloカタカナ세계"}}
        self.assertEqual(lang_id_route_decision(window), "mixed")

    def test_low_entropy_routes_separated(self) -> None:
        window = {"separated_text_per_speaker": {"a": "你好世界你好"}}
        self.assertEqual(lang_id_route_decision(window), "separated")

    def test_empty_window_routes_separated(self) -> None:
        window = {"separated_text_per_speaker": {}}
        self.assertEqual(lang_id_route_decision(window), "separated")


# ======================================================================
# cpwer_for (RQ58 helper, reused)
# ======================================================================
class TestCpwerFor(unittest.TestCase):
    def test_mixed_returns_mixed_cpwer(self) -> None:
        window = {"always_mixed_cpwer": 1.5, "always_separated_cpwer": 2.0}
        self.assertEqual(cpwer_for(window, "mixed"), 1.5)

    def test_separated_returns_separated_cpwer(self) -> None:
        window = {"always_mixed_cpwer": 1.5, "always_separated_cpwer": 2.0}
        self.assertEqual(cpwer_for(window, "separated"), 2.0)

    def test_returns_float(self) -> None:
        window = {"always_mixed_cpwer": "1.0", "always_separated_cpwer": "2.0"}
        self.assertIsInstance(cpwer_for(window, "mixed"), float)


# ======================================================================
# kl_flag (NEW RQ60 helper)
# ======================================================================
class TestKlFlag(unittest.TestCase):
    def test_above_threshold_flags(self) -> None:
        self.assertTrue(kl_flag(10.0, 5.42))

    def test_below_threshold_no_flag(self) -> None:
        self.assertFalse(kl_flag(1.0, 5.42))

    def test_at_threshold_flags(self) -> None:
        self.assertTrue(kl_flag(5.42, 5.42))


# ======================================================================
# lang_id_flag (NEW RQ60 helper)
# ======================================================================
class TestLangIdFlag(unittest.TestCase):
    def test_above_threshold_flags(self) -> None:
        self.assertTrue(lang_id_flag(0.5, 0.38))

    def test_below_threshold_no_flag(self) -> None:
        self.assertFalse(lang_id_flag(0.1, 0.38))

    def test_at_threshold_flags(self) -> None:
        self.assertTrue(lang_id_flag(0.38, 0.38))


# ======================================================================
# or_flag (NEW RQ60 ensemble helper)
# ======================================================================
class TestOrFlag(unittest.TestCase):
    def test_both_flag_returns_true(self) -> None:
        self.assertTrue(or_flag(10.0, 5.42, 0.5, 0.38))

    def test_kl_only_flag_returns_true(self) -> None:
        self.assertTrue(or_flag(10.0, 5.42, 0.1, 0.38))

    def test_lang_id_only_flag_returns_true(self) -> None:
        self.assertTrue(or_flag(1.0, 5.42, 0.5, 0.38))

    def test_neither_flag_returns_false(self) -> None:
        self.assertFalse(or_flag(1.0, 5.42, 0.1, 0.38))

    def test_mode_s_scenario_kl_catches(self) -> None:
        # Mode S: high KL, low lang-id → OR catches via KL.
        self.assertTrue(or_flag(13.0, 5.42, 0.05, 0.38))

    def test_diverse_scenario_lang_id_catches(self) -> None:
        # Diverse hallucination: low KL, high lang-id → OR catches via lang-id.
        self.assertTrue(or_flag(1.0, 5.42, 1.5, 0.38))


# ======================================================================
# and_flag (NEW RQ60 ensemble helper)
# ======================================================================
class TestAndFlag(unittest.TestCase):
    def test_both_flag_returns_true(self) -> None:
        self.assertTrue(and_flag(10.0, 5.42, 0.5, 0.38))

    def test_kl_only_flag_returns_false(self) -> None:
        self.assertFalse(and_flag(10.0, 5.42, 0.1, 0.38))

    def test_lang_id_only_flag_returns_false(self) -> None:
        self.assertFalse(and_flag(1.0, 5.42, 0.5, 0.38))

    def test_neither_flag_returns_false(self) -> None:
        self.assertFalse(and_flag(1.0, 5.42, 0.1, 0.38))

    def test_mode_s_scenario_and_misses(self) -> None:
        # Mode S: high KL, low lang-id → AND misses (needs both).
        self.assertFalse(and_flag(13.0, 5.42, 0.05, 0.38))

    def test_diverse_scenario_and_misses(self) -> None:
        # Diverse: low KL, high lang-id → AND misses.
        self.assertFalse(and_flag(1.0, 5.42, 1.5, 0.38))


# ======================================================================
# or_route_decision (NEW RQ60 ensemble helper)
# ======================================================================
class TestOrRouteDecision(unittest.TestCase):
    def test_both_flag_routes_mixed(self) -> None:
        self.assertEqual(or_route_decision(10.0, 5.42, 0.5, 0.38), "mixed")

    def test_kl_only_routes_mixed(self) -> None:
        self.assertEqual(or_route_decision(10.0, 5.42, 0.1, 0.38), "mixed")

    def test_lang_id_only_routes_mixed(self) -> None:
        self.assertEqual(or_route_decision(1.0, 5.42, 0.5, 0.38), "mixed")

    def test_neither_routes_separated(self) -> None:
        self.assertEqual(or_route_decision(1.0, 5.42, 0.1, 0.38), "separated")

    def test_or_is_superset_of_kl_alone(self) -> None:
        # OR flags at least every window KL-alone flags.
        for ks in [0.0, 3.0, 5.42, 6.0, 13.0]:
            for ls in [0.0, 0.2, 0.38, 0.5, 1.5]:
                if kl_flag(ks, 5.42):
                    self.assertEqual(
                        or_route_decision(ks, 5.42, ls, 0.38), "mixed"
                    )

    def test_or_is_superset_of_lang_id_alone(self) -> None:
        for ks in [0.0, 3.0, 5.42, 6.0, 13.0]:
            for ls in [0.0, 0.2, 0.38, 0.5, 1.5]:
                if lang_id_flag(ls, 0.38):
                    self.assertEqual(
                        or_route_decision(ks, 5.42, ls, 0.38), "mixed"
                    )


# ======================================================================
# and_route_decision (NEW RQ60 ensemble helper)
# ======================================================================
class TestAndRouteDecision(unittest.TestCase):
    def test_both_flag_routes_mixed(self) -> None:
        self.assertEqual(and_route_decision(10.0, 5.42, 0.5, 0.38), "mixed")

    def test_kl_only_routes_separated(self) -> None:
        self.assertEqual(and_route_decision(10.0, 5.42, 0.1, 0.38), "separated")

    def test_lang_id_only_routes_separated(self) -> None:
        self.assertEqual(and_route_decision(1.0, 5.42, 0.5, 0.38), "separated")

    def test_neither_routes_separated(self) -> None:
        self.assertEqual(and_route_decision(1.0, 5.42, 0.1, 0.38), "separated")

    def test_and_is_subset_of_kl_alone(self) -> None:
        # AND flags at most every window KL-alone flags.
        for ks in [0.0, 3.0, 5.42, 6.0, 13.0]:
            for ls in [0.0, 0.2, 0.38, 0.5, 1.5]:
                if and_route_decision(ks, 5.42, ls, 0.38) == "mixed":
                    self.assertEqual(kl_route_decision(ks, 5.42), "mixed")

    def test_and_is_subset_of_lang_id_alone(self) -> None:
        for ks in [0.0, 3.0, 5.42, 6.0, 13.0]:
            for ls in [0.0, 0.2, 0.38, 0.5, 1.5]:
                if and_route_decision(ks, 5.42, ls, 0.38) == "mixed":
                    self.assertTrue(lang_id_flag(ls, 0.38))


# ======================================================================
# bootstrap_indices (RQ39 helper)
# ======================================================================
class TestBootstrapIndices(unittest.TestCase):
    def test_shape(self) -> None:
        idx = bootstrap_indices(10, 100, 42)
        self.assertEqual(idx.shape, (100, 10))

    def test_values_in_range(self) -> None:
        idx = bootstrap_indices(10, 100, 42)
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < 10))

    def test_deterministic(self) -> None:
        idx1 = bootstrap_indices(10, 50, 42)
        idx2 = bootstrap_indices(10, 50, 42)
        np.testing.assert_array_equal(idx1, idx2)


# ======================================================================
# bootstrap_distribution (RQ39 helper)
# ======================================================================
class TestBootstrapDistribution(unittest.TestCase):
    def test_mean_close_to_sample_mean(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        boot = bootstrap_distribution(values, 10000, 42)
        self.assertAlmostEqual(boot.mean(), values.mean(), places=1)

    def test_length_matches_n_boot(self) -> None:
        boot = bootstrap_distribution(np.array([1.0, 2.0]), 100, 42)
        self.assertEqual(len(boot), 100)

    def test_deterministic(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        b1 = bootstrap_distribution(values, 100, 42)
        b2 = bootstrap_distribution(values, 100, 42)
        np.testing.assert_array_equal(b1, b2)


# ======================================================================
# percentile_ci (RQ39 helper)
# ======================================================================
class TestPercentileCi(unittest.TestCase):
    def test_ci_brackets_mean(self) -> None:
        boot = np.random.default_rng(42).normal(1.0, 0.1, size=10000)
        lo, hi = percentile_ci(boot)
        self.assertLessEqual(lo, 1.0)
        self.assertGreaterEqual(hi, 1.0)

    def test_lo_less_than_hi(self) -> None:
        boot = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        lo, hi = percentile_ci(boot)
        self.assertLess(lo, hi)

    def test_alpha_05_uses_2_5_97_5_percentiles(self) -> None:
        boot = np.arange(1, 101, dtype=float)
        lo, hi = percentile_ci(boot, alpha=0.05)
        self.assertAlmostEqual(lo, 3.0, delta=1.0)
        self.assertAlmostEqual(hi, 98.0, delta=1.0)


# ======================================================================
# _jackknife_means (RQ39 helper)
# ======================================================================
class TestJackknifeMeans(unittest.TestCase):
    def test_length_matches_input(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        jack = _jackknife_means(values)
        self.assertEqual(len(jack), 4)

    def test_jackknife_mean_formula(self) -> None:
        # Leave-one-out mean of [1,2,3,4] leaving out 1 = (2+3+4)/3 = 3.0
        values = np.array([1.0, 2.0, 3.0, 4.0])
        jack = _jackknife_means(values)
        self.assertAlmostEqual(jack[0], 3.0, places=9)
        self.assertAlmostEqual(jack[1], 8.0 / 3.0, places=9)

    def test_single_element(self) -> None:
        jack = _jackknife_means(np.array([5.0]))
        self.assertEqual(len(jack), 1)


# ======================================================================
# bca_ci (RQ39 helper)
# ======================================================================
class TestBcaCi(unittest.TestCase):
    def test_bca_ci_brackets_mean(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(1.0, 0.2, size=77)
        boot = bootstrap_distribution(values, 10000, 42)
        lo, hi = bca_ci(values, boot)
        self.assertLessEqual(lo, values.mean())
        self.assertGreaterEqual(hi, values.mean())

    def test_bca_ci_lo_less_than_hi(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(1.0, 0.2, size=50)
        boot = bootstrap_distribution(values, 5000, 42)
        lo, hi = bca_ci(values, boot)
        self.assertLess(lo, hi)

    def test_bca_ci_constant_data(self) -> None:
        values = np.array([1.0] * 10)
        boot = np.array([1.0] * 100)
        lo, hi = bca_ci(values, boot)
        self.assertAlmostEqual(lo, 1.0, places=6)
        self.assertAlmostEqual(hi, 1.0, places=6)

    def test_bca_ci_single_element(self) -> None:
        values = np.array([5.0])
        boot = np.array([5.0])
        lo, hi = bca_ci(values, boot)
        self.assertAlmostEqual(lo, 5.0, places=6)
        self.assertAlmostEqual(hi, 5.0, places=6)


# ======================================================================
# paired_delta_distribution (RQ39 helper)
# ======================================================================
class TestPairedDeltaDistribution(unittest.TestCase):
    def test_mean_close_to_delta_of_means(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        dist = paired_delta_distribution(a, b, 10000, 42)
        self.assertAlmostEqual(dist.mean(), 0.5, places=1)

    def test_length_matches_n_boot(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([0.5, 1.5])
        dist = paired_delta_distribution(a, b, 100, 42)
        self.assertEqual(len(dist), 100)

    def test_shape_mismatch_raises(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            paired_delta_distribution(a, b, 100, 42)


# ======================================================================
# paired_delta_ci (RQ39 helper)
# ======================================================================
class TestPairedDeltaCi(unittest.TestCase):
    def test_ci_brackets_point_delta(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        lo, hi = paired_delta_ci(a, b, 10000, 42)
        delta = a.mean() - b.mean()
        self.assertLessEqual(lo, delta + 0.01)
        self.assertGreaterEqual(hi, delta - 0.01)

    def test_lo_less_than_hi(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.5, 2.5])
        lo, hi = paired_delta_ci(a, b, 1000, 42)
        self.assertLess(lo, hi)


# ======================================================================
# calibrate_threshold_at_specificity (RQ34 primitive)
# ======================================================================
class TestCalibrateThreshold(unittest.TestCase):
    def test_picks_smallest_threshold_meeting_spec(self) -> None:
        neg = [0.1, 0.2, 0.3, 0.4, 0.5]
        pos = [0.8, 0.9]
        cal = calibrate_threshold_at_specificity(neg, pos, 0.8)
        self.assertLessEqual(cal["specificity"], 1.0)
        self.assertGreaterEqual(cal["specificity"], 0.8)

    def test_no_neg_returns_inf(self) -> None:
        cal = calibrate_threshold_at_specificity([], [1.0], 0.9)
        self.assertEqual(cal["threshold"], float("inf"))


# ======================================================================
# evaluate_at_threshold (RQ34 primitive)
# ======================================================================
class TestEvaluateAtThreshold(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        scores = [1.0, 2.0, 8.0, 9.0]
        labels = [0, 0, 1, 1]
        res = evaluate_at_threshold(scores, labels, 5.0)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertAlmostEqual(res["sensitivity"], 1.0)
        self.assertAlmostEqual(res["specificity"], 1.0)

    def test_counts(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [0, 0, 1, 1]
        res = evaluate_at_threshold(scores, labels, 2.5)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["tn"], 2)
        self.assertEqual(res["fn"], 0)


# ======================================================================
# subgroup_sensitivity (RQ34 primitive)
# ======================================================================
class TestSubgroupSensitivity(unittest.TestCase):
    def test_all_flagged(self) -> None:
        scores = [10.0, 12.0]
        mask = [True, True]
        res = subgroup_sensitivity(scores, mask, 5.0)
        self.assertAlmostEqual(res["sensitivity"], 1.0)
        self.assertEqual(res["tp"], 2)

    def test_none_flagged(self) -> None:
        scores = [1.0, 2.0]
        mask = [True, True]
        res = subgroup_sensitivity(scores, mask, 5.0)
        self.assertAlmostEqual(res["sensitivity"], 0.0)

    def test_empty_subgroup(self) -> None:
        res = subgroup_sensitivity([1.0], [False], 5.0)
        self.assertEqual(res["n"], 0)


# ======================================================================
# Integration: ensemble routing on synthetic data
# ======================================================================
class TestEnsembleIntegration(unittest.TestCase):
    """End-to-end ensemble routing on synthetic windows."""

    def _make_windows(self) -> list[dict]:
        """3 synthetic windows: clean, mode-s-like, diverse-like."""
        return [
            # Clean: low KL, low lang-id → all routers choose separated.
            {
                "window_id": 0,
                "always_mixed_cpwer": 1.0,
                "always_separated_cpwer": 1.0,
                "oracle_best_cpwer": 1.0,
                "separated_text_per_speaker": {"a": "你好世界大家好"},
            },
            # Mode-S-like: high KL, low lang-id → KL catches, lang-id misses.
            {
                "window_id": 1,
                "always_mixed_cpwer": 1.0,
                "always_separated_cpwer": 2.0,
                "oracle_best_cpwer": 1.0,
                "separated_text_per_speaker": {"a": "重複重複重複重複重複重複"},
            },
            # Diverse-like: low KL, high lang-id → lang-id catches.
            {
                "window_id": 2,
                "always_mixed_cpwer": 1.0,
                "always_separated_cpwer": 2.0,
                "oracle_best_cpwer": 1.0,
                "separated_text_per_speaker": {"a": "Helloカタカナ세계 мир"},
            },
        ]

    def test_or_ensemble_catches_both_failure_types(self) -> None:
        windows = self._make_windows()
        # KL catches window 1 (mode-s-like), lang-id catches window 2 (diverse).
        # OR catches both → all 3 route correctly.
        decisions = []
        for w in windows:
            ent = max_across_speakers(w, language_id_entropy)
            # Use a low KL threshold so mode-s-like is caught
            dec = or_route_decision(10.0 if w["window_id"] == 1 else 1.0,
                                    5.42, ent, 0.38)
            decisions.append(dec)
        self.assertEqual(decisions[0], "separated")
        self.assertEqual(decisions[1], "mixed")
        self.assertEqual(decisions[2], "mixed")

    def test_and_ensemble_misses_single_signal_windows(self) -> None:
        windows = self._make_windows()
        decisions = []
        for w in windows:
            ent = max_across_speakers(w, language_id_entropy)
            dec = and_route_decision(10.0 if w["window_id"] == 1 else 1.0,
                                     5.42, ent, 0.38)
            decisions.append(dec)
        # AND requires both signals; mode-s-like has low lang-id → separated.
        self.assertEqual(decisions[1], "separated")
        # Diverse-like has low KL → separated.
        self.assertEqual(decisions[2], "separated")

    def test_or_cpwer_le_oracle(self) -> None:
        """OR ensemble cpWER >= oracle (can't beat per-window oracle)."""
        windows = self._make_windows()
        kl_scores = [1.0, 10.0, 1.0]
        or_cpwer = 0.0
        oracle_cpwer = 0.0
        for w, ks in zip(windows, kl_scores):
            ent = max_across_speakers(w, language_id_entropy)
            dec = or_route_decision(ks, 5.42, ent, 0.38)
            or_cpwer += cpwer_for(w, dec)
            oracle_cpwer += float(w["oracle_best_cpwer"])
        self.assertGreaterEqual(or_cpwer / 3, oracle_cpwer / 3 - EPS)

    def test_or_cpwer_le_kl_alone(self) -> None:
        """OR ensemble cpWER <= KL-alone (OR is a superset of KL flags)."""
        windows = self._make_windows()
        kl_scores = [1.0, 10.0, 1.0]
        or_cpwer = 0.0
        kl_cpwer = 0.0
        for w, ks in zip(windows, kl_scores):
            ent = max_across_speakers(w, language_id_entropy)
            or_dec = or_route_decision(ks, 5.42, ent, 0.38)
            kl_dec = kl_route_decision(ks, 5.42)
            or_cpwer += cpwer_for(w, or_dec)
            kl_cpwer += cpwer_for(w, kl_dec)
        # OR routes at least as many to mixed as KL-alone.
        # When mixed_cpwer <= separated_cpwer (hallucination case), OR <= KL.
        self.assertLessEqual(or_cpwer, kl_cpwer + EPS)


if __name__ == "__main__":
    unittest.main()
