"""RQ60: KL+lang-id ensemble corrected router.

REANALYSIS ONLY — no Whisper / no ASR model / no LLM is run. This script reads
the existing AISHELL-4 external-validation results
(``results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json``,
label ``external/sanity-check``, PR #890) and tests whether an ensemble of the
two complementary detectors beats both individually:

  * **KL detector** (RQ58, PR #981): character 2-gram KL-divergence anomaly
    detector. Catches Mode S (monoscript-Chinese near-duplicate hallucinations,
    windows 22 & 30) at 100% sensitivity — the FIRST detector to do so — but
    has 4 false positives. KL-alone cpWER = 1.030303.
  * **lang-id entropy** (RQ13/RQ16, PR #912): Shannon entropy over Unicode
    script categories. Catches 94.6% of all hallucinations (diverse
    multilingual gibberish) but 0% of Mode S. lang-id-alone cpWER = 1.04329.

The two detectors are COMPLEMENTARY: KL catches Mode S (which lang-id misses),
lang-id catches diverse hallucinations (which KL also catches). RQ60 asks: does
an OR/AND ensemble of KL and lang-id beat both individually?

Method
------
For each of the 77 AISHELL-4 windows:

  1. Compute the KL score (RQ58's MAX-across-speakers 2-gram KL, threshold
     5.418144 — RQ58's empirically-calibrated threshold at >=90% specificity).
  2. Compute the lang-id entropy (RQ13's MAX-across-speakers, threshold 0.38).
     NOTE: threshold 0.38 gives IDENTICAL results to RQ16's 0.409 (no score
     falls between them); both reproduce RQ16's 1.043290 with 3/40 FPs.
  3. **OR ensemble**: route to MIXED if KL >= 5.418144 OR lang_id >= 0.38.
     The OR ensemble is a superset of each detector's flags — it catches
     everything either detector catches.
  4. **AND ensemble**: route to MIXED if KL >= 5.418144 AND lang_id >= 0.38.
     The AND ensemble is the intersection — it flags only windows where both
     detectors agree, trading sensitivity for fewer false positives.
  5. Per-window cpWER is the chosen route's stored word-level cpWER
     (``always_mixed_cpwer`` / ``always_separated_cpwer``) — matches RQ16/RQ58
     bit-for-bit.
  6. Bootstrap (B=10000, seed=42) the OR and AND ensemble cpWER. Report
     percentile + BCa CIs (RQ39's framework).

Hypotheses
----------
- H60a: OR ensemble cpWER < 1.030 (beats KL-alone, the current best). KILLED
  if cpWER >= 1.030.
- H60b: OR ensemble catches 100% Mode S (from KL) AND >= 90% all
  hallucinations. KILLED if < 100% Mode S OR < 90% all.
- H60c: AND ensemble FP rate < 7.5% (3/40). KILLED if AND FP >= 7.5%.

This script is pure reanalysis (numpy + scipy + stdlib only; no Whisper / no
LLM / no ollama). The KL detector primitives are imported VERBATIM from
``src.llm_semantic_critic`` (RQ34) so the detector is directly comparable to
RQ58. The BCa CI helpers are reimplemented from RQ39
(``results/frontier/bootstrap_ci_corrected_router/bootstrap_ci_analysis.py``)
so the CI methodology matches.

Label: experimental/frontier. Closes #984.

Run:
    /opt/homebrew/bin/python3 results/frontier/ensemble_corrected_router/ensemble_router_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SRC_JSON = (
    PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "ensemble_router_results.csv"
OUT_JSON = OUT_DIR / "ensemble_router_results.json"
FINDINGS_MD = OUT_DIR / "FINDINGS.md"

# ---------------------------------------------------------- import RQ34 primitives
# KL detector + labelling primitives lifted VERBATIM from src.llm_semantic_critic
# (RQ34, PR #951) so the detector is directly comparable to RQ58.
from src.llm_semantic_critic import (  # noqa: E402
    CATASTROPHIC_CPWER,
    EPS,
    N_BOOT,
    SEED,
    TARGET_SPECIFICITY,
    average_distributions,
    build_reference_distribution,
    calibrate_threshold_at_specificity,
    char_distribution,
    char_ngrams,
    compute_anomaly_score,
    evaluate_at_threshold,
    kl_divergence,
    label_window,
    language_id_entropy,
    max_across_speakers,
    separated_concat,
    subgroup_sensitivity,
)

# ------------------------------------------------------------------ config
# 2-gram (bigram) character n-gram, per the RQ58/RQ60 task spec.
N_GRAM = 2

# RQ58's empirically-calibrated KL threshold at >=90% specificity (PR #981).
# The task spec says "5.42" (rounded); we use the precise value 5.418144 for
# exact reproducibility of RQ58's KL-alone cpWER 1.030303.
KL_THRESHOLD = 5.418144

# lang-id entropy threshold (task spec pre-registers 0.38). This gives IDENTICAL
# results to RQ16's 0.409 (no lang-id score falls between 0.38 and 0.409); both
# reproduce RQ16's 1.043290 with 3/40 FPs and 94.6% sensitivity.
LANG_ID_THRESHOLD = 0.38

# RQ58 / RQ16 / baseline reference values for hypothesis comparison (word-level).
RQ58_KL_CPWER = 1.030303          # RQ58's KL-alone (PR #981)
RQ16_LANG_ID_CPWER = 1.04329      # RQ16's lang-id-alone (PR #912)
ALWAYS_MIXED_CPWER = 1.17316
ALWAYS_SEPARATED_CPWER = 1.590909
ROUTER_V2_CPWER = 1.205628
ORACLE_BEST_CPWER = 1.017316

# H60a kill threshold (task spec: "KILLED if >= 1.030").
H60A_KILL_THRESHOLD = 1.030

# H60b kill thresholds.
H60B_MODE_S_SENS_KILL = 1.0       # < 100% Mode S → KILLED
H60B_ALL_SENS_KILL = 0.90         # < 90% all halluc → KILLED

# H60c kill threshold: AND FP rate >= 7.5% (3/40) → KILLED.
AND_FLAG_FP_KILL_RATE = 0.075

ALPHA = 0.05

# Mode S windows (RQ19 verified): monoscript-Chinese near-duplicate
# hallucinations that escape every surface detector.
MODE_S_WINDOW_IDS = {22, 30}


# ===========================================================================
# Part 1: KL + lang-id detector primitives (reused from RQ58/RQ34/RQ13)
# ===========================================================================
def build_kl_reference(
    labels: list[dict[str, Any]], n: int = N_GRAM,
) -> dict[str, float]:
    """Build the reference character n-gram distribution from the non-hallucinated
    tracks' separated text (RQ58 verbatim).

    Mirrors RQ34's ``run_ngram_fallback``: the reference is the average n-gram
    distribution of the 40 non-hallucinated windows' concatenated separated
    text (``separated_concat``)."""
    neg_texts = [lbl["separated_text"] for lbl in labels if not lbl["hallucinated"]]
    return build_reference_distribution(neg_texts, n=n)


def compute_kl_scores(
    windows: list[dict[str, Any]],
    ref_distribution: dict[str, float],
    n: int = N_GRAM,
    eps: float = EPS,
) -> list[float]:
    """MAX-across-speakers n-gram KL-divergence anomaly score per window (RQ58 verbatim).

    For each window, apply RQ34's ``compute_anomaly_score`` to each non-empty
    per-speaker separated transcript, then take the MAX (the RQ12/RQ13
    worst-case-track convention). Returns 0.0 for windows with no non-empty
    speaker text."""
    return [
        max_across_speakers(
            w, lambda t: compute_anomaly_score(t, ref_distribution, n=n, eps=eps)
        )
        for w in windows
    ]


def compute_lang_id_scores(windows: list[dict[str, Any]]) -> list[float]:
    """MAX-across-speakers lang-id entropy per window (RQ13/RQ16 verbatim).

    For each window, apply RQ13's ``language_id_entropy`` to each non-empty
    per-speaker separated transcript, then take the MAX (worst-case track)."""
    return [max_across_speakers(w, language_id_entropy) for w in windows]


# ===========================================================================
# Part 2: flag + route decision helpers (RQ58 reused + NEW RQ60 ensemble)
# ===========================================================================
def kl_flag(kl_score: float, kl_threshold: float, eps: float = EPS) -> bool:
    """True if the KL score flags the window as hallucination-likely.

    Uses ``>= threshold - eps`` to match RQ34's ``evaluate_at_threshold`` and
    RQ58's ``kl_route_decision`` flag convention."""
    return kl_score >= kl_threshold - eps


def lang_id_flag(
    lang_id_score: float, lang_id_threshold: float, eps: float = EPS,
) -> bool:
    """True if the lang-id entropy flags the window as hallucination-likely.

    Uses ``>= threshold - eps`` (matching the KL convention). NOTE: RQ16 used
    strict ``> 0.409``; since no lang-id score equals exactly 0.38 or 0.409,
    ``>=`` and ``>`` give identical results here."""
    return lang_id_score >= lang_id_threshold - eps


def or_flag(
    kl_score: float,
    kl_threshold: float,
    lang_id_score: float,
    lang_id_threshold: float,
    eps: float = EPS,
) -> bool:
    """OR ensemble flag: True if EITHER detector flags (superset of each).

    The OR ensemble catches everything either detector catches. It maximises
    sensitivity at the cost of potentially more false positives."""
    return kl_flag(kl_score, kl_threshold, eps) or lang_id_flag(
        lang_id_score, lang_id_threshold, eps
    )


def and_flag(
    kl_score: float,
    kl_threshold: float,
    lang_id_score: float,
    lang_id_threshold: float,
    eps: float = EPS,
) -> bool:
    """AND ensemble flag: True if BOTH detectors flag (intersection of each).

    The AND ensemble flags only windows where both detectors agree. It
    minimises false positives at the cost of sensitivity (misses windows where
    only one detector fires — e.g. Mode S, which has high KL but low lang-id)."""
    return kl_flag(kl_score, kl_threshold, eps) and lang_id_flag(
        lang_id_score, lang_id_threshold, eps
    )


def kl_route_decision(kl_score: float, threshold: float, eps: float = EPS) -> str:
    """RQ58's KL-alone route decision: MIXED if KL >= threshold, else SEPARATED.

    Uses ``>= threshold - eps`` to match RQ34's ``evaluate_at_threshold`` flag
    convention."""
    return "mixed" if kl_score >= threshold - eps else "separated"


def lang_id_route_decision(window: dict[str, Any]) -> str:
    """RQ16's lang-id corrected-router decision (for comparison / sanity check).

    Route to MIXED if ``max_across_speakers(separated, language_id_entropy) >=
    LANG_ID_THRESHOLD`` bits, else SEPARATED. RQ39 verified lang-id alone is
    cpWER-identical to RQ16's full three-guard corrected router on AISHELL-4
    (1.04329)."""
    ent = max_across_speakers(window, language_id_entropy)
    return "mixed" if ent >= LANG_ID_THRESHOLD - EPS else "separated"


def or_route_decision(
    kl_score: float,
    kl_threshold: float,
    lang_id_score: float,
    lang_id_threshold: float,
    eps: float = EPS,
) -> str:
    """OR ensemble route: MIXED if KL >= kl_threshold OR lang_id >= lang_id_threshold."""
    return "mixed" if or_flag(
        kl_score, kl_threshold, lang_id_score, lang_id_threshold, eps
    ) else "separated"


def and_route_decision(
    kl_score: float,
    kl_threshold: float,
    lang_id_score: float,
    lang_id_threshold: float,
    eps: float = EPS,
) -> str:
    """AND ensemble route: MIXED if KL >= kl_threshold AND lang_id >= lang_id_threshold."""
    return "mixed" if and_flag(
        kl_score, kl_threshold, lang_id_score, lang_id_threshold, eps
    ) else "separated"


def cpwer_for(window: dict[str, Any], choice: str) -> float:
    """Stored word-level cpWER for the chosen route (matches RQ16/RQ58 bit-for-bit).

    ``choice`` = "mixed" -> ``always_mixed_cpwer``; "separated" ->
    ``always_separated_cpwer``."""
    return float(
        window["always_mixed_cpwer"] if choice == "mixed"
        else window["always_separated_cpwer"]
    )


# ===========================================================================
# Part 3: bootstrap + BCa CI helpers (reimplemented from RQ39, PURE)
# ===========================================================================
def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Return an ``(n_boot, n)`` int array of resample indices (with replacement).

    Same convention as RQ16/RQ39: ``rng.integers(0, n, size=n)`` per resample.
    Deterministic for a fixed ``seed``."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_distribution(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Return an ``n_boot`` array of bootstrap means of ``values``.

    Resamples ``values`` with replacement (``n`` indices per resample) and takes
    the mean. Deterministic for a fixed ``seed``."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = bootstrap_indices(n, n_boot, seed)
    return values[idx].mean(axis=1)


def percentile_ci(
    boot_dist: np.ndarray, alpha: float = ALPHA,
) -> tuple[float, float]:
    """Percentile CI: ``(100*alpha/2, 100*(1-alpha/2))`` percentiles of the
    bootstrap distribution. Returns ``(lo, hi)``."""
    boot_dist = np.asarray(boot_dist, dtype=float)
    lo = float(np.percentile(boot_dist, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_dist, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _jackknife_means(values: np.ndarray) -> np.ndarray:
    """Leave-one-out jackknife means of ``values`` (length-``n`` array).

    O(n) via the identity: mean of n-1 values = (n*mean - x_i) / (n-1).
    Used by ``bca_ci`` to compute the acceleration ``a``."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.array([float(values.mean())])
    total = float(values.sum())
    return (total - values) / (n - 1)


def bca_ci(
    values: np.ndarray, boot_dist: np.ndarray, alpha: float = ALPHA,
) -> tuple[float, float]:
    """BCa (bias-corrected + accelerated) CI for the mean of ``values``.

    Standard Efron & Tibshirani BCa formula (reimplemented from RQ39 verbatim).
    Edge cases (constant data, zero denominator, ``P(boot < theta_hat)`` of 0/1)
    are handled by clipping to a small epsilon and falling back to the
    percentile CI when the acceleration is undefined."""
    values = np.asarray(values, dtype=float)
    boot_dist = np.asarray(boot_dist, dtype=float)
    n = len(values)
    if n < 2:
        theta = float(values.mean()) if n == 1 else float("nan")
        return theta, theta

    theta_hat = float(values.mean())

    # --- bias correction z0
    prop_less = float(np.mean(boot_dist < theta_hat))
    eps_clip = 0.5 / len(boot_dist)
    prop_less = min(max(prop_less, eps_clip), 1.0 - eps_clip)
    z0 = float(norm.ppf(prop_less))

    # --- acceleration a via jackknife
    jack = _jackknife_means(values)
    jack_mean = float(jack.mean())
    diff = jack_mean - jack
    num = float(np.sum(diff ** 3))
    den = 6.0 * (float(np.sum(diff ** 2)) ** 1.5)
    a = num / den if den > 0 else 0.0

    # --- BCa alpha bounds
    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))

    denom_lo = 1.0 - a * (z0 + z_lo)
    denom_hi = 1.0 - a * (z0 + z_hi)
    if abs(denom_lo) < EPS or abs(denom_hi) < EPS:
        return percentile_ci(boot_dist, alpha)

    alpha1 = float(norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha2 = float(norm.cdf(z0 + (z0 + z_hi) / denom_hi))

    alpha1 = min(max(alpha1, 0.0), 1.0)
    alpha2 = min(max(alpha2, 0.0), 1.0)

    lo = float(np.percentile(boot_dist, 100.0 * alpha1))
    hi = float(np.percentile(boot_dist, 100.0 * alpha2))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def paired_delta_distribution(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int,
) -> np.ndarray:
    """Bootstrap distribution of ``mean(a[idx]) - mean(b[idx])`` (paired).

    Same resample indices for both ``a`` and ``b`` (paired design). Returns an
    ``n_boot`` array. Deterministic for a fixed ``seed``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired_delta_distribution: a and b must have the same shape, got "
            f"{a.shape} vs {b.shape}"
        )
    n = len(a)
    idx = bootstrap_indices(n, n_boot, seed)
    return a[idx].mean(axis=1) - b[idx].mean(axis=1)


def paired_delta_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, alpha: float = ALPHA,
) -> tuple[float, float]:
    """Percentile CI for the paired bootstrap ``mean(a) - mean(b)``."""
    dist = paired_delta_distribution(a, b, n_boot, seed)
    return percentile_ci(dist, alpha)


# ===========================================================================
# Part 4: driver
# ===========================================================================
def _round6(x: float) -> float:
    return round(float(x), 6)


def _ci_pair(ci: tuple[float, float]) -> list[float]:
    return [_round6(ci[0]), _round6(ci[1])]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    windows = data["windows"]
    n = len(windows)

    # --- per-window labels (RQ34 verbatim) + surface features
    labels = [label_window(w) for w in windows]
    n_halluc = sum(1 for l in labels if l["hallucinated"])
    n_nonhalluc = n - n_halluc
    n_mode_s = sum(1 for l in labels if l["mode_s"])
    n_diverse = sum(1 for l in labels if l["diverse_hallucination"])
    mode_s_ids = [l["window_id"] for l in labels if l["mode_s"]]

    # --- build 2-gram reference distribution from 40 non-hallucinated tracks
    ref_dist = build_kl_reference(labels, n=N_GRAM)
    ref_vocab_size = len(ref_dist)

    # --- MAX-across-speakers 2-gram KL scores (RQ58)
    kl_scores = compute_kl_scores(windows, ref_dist, n=N_GRAM)

    # --- MAX-across-speakers lang-id entropy scores (RQ13)
    lang_id_scores = compute_lang_id_scores(windows)

    # --- verify KL-alone reproduces RQ58 (sanity check)
    kl_alone_cpwer = float(np.mean([
        cpwer_for(w, kl_route_decision(ks, KL_THRESHOLD))
        for w, ks in zip(windows, kl_scores)
    ]))

    # --- verify lang-id-alone reproduces RQ16 (sanity check)
    lang_id_alone_cpwer = float(np.mean([
        cpwer_for(w, lang_id_route_decision(w))
        for w in windows
    ]))

    # --- per-window routing + cpWER for OR and AND ensembles
    rows: list[dict[str, Any]] = []
    for w, lbl, ks, ls in zip(windows, labels, kl_scores, lang_id_scores):
        kl_dec = kl_route_decision(ks, KL_THRESHOLD)
        kl_cpwer = cpwer_for(w, kl_dec)
        lang_dec = lang_id_route_decision(w)
        lang_cpwer = cpwer_for(w, lang_dec)
        or_dec = or_route_decision(ks, KL_THRESHOLD, ls, LANG_ID_THRESHOLD)
        or_cpwer = cpwer_for(w, or_dec)
        and_dec = and_route_decision(ks, KL_THRESHOLD, ls, LANG_ID_THRESHOLD)
        and_cpwer = cpwer_for(w, and_dec)
        rows.append({
            "window_id": w["window_id"],
            "overlap_label": w["overlap_label"],
            "hallucinated": int(lbl["hallucinated"]),
            "mode_s": int(lbl["mode_s"]),
            "diverse_hallucination": int(lbl["diverse_hallucination"]),
            "lang_id_entropy": _round6(ls),
            "kl_score": _round6(ks),
            "always_mixed_cpwer": _round6(float(w["always_mixed_cpwer"])),
            "always_separated_cpwer": _round6(float(w["always_separated_cpwer"])),
            "router_v2_cpwer": _round6(float(w["router_v2_cpwer"])),
            "oracle_best_cpwer": _round6(float(w["oracle_best_cpwer"])),
            "kl_flag": int(kl_flag(ks, KL_THRESHOLD)),
            "lang_id_flag": int(lang_id_flag(ls, LANG_ID_THRESHOLD)),
            "or_flag": int(or_flag(ks, KL_THRESHOLD, ls, LANG_ID_THRESHOLD)),
            "and_flag": int(and_flag(ks, KL_THRESHOLD, ls, LANG_ID_THRESHOLD)),
            "kl_decision": kl_dec,
            "kl_cpwer": _round6(kl_cpwer),
            "lang_id_decision": lang_dec,
            "lang_id_cpwer": _round6(lang_cpwer),
            "or_decision": or_dec,
            "or_cpwer": _round6(or_cpwer),
            "and_decision": and_dec,
            "and_cpwer": _round6(and_cpwer),
        })

    # --- aggregates
    kl_arr = np.array([r["kl_cpwer"] for r in rows], dtype=float)
    lang_arr = np.array([r["lang_id_cpwer"] for r in rows], dtype=float)
    or_arr = np.array([r["or_cpwer"] for r in rows], dtype=float)
    and_arr = np.array([r["and_cpwer"] for r in rows], dtype=float)
    mixed_arr = np.array([r["always_mixed_cpwer"] for r in rows], dtype=float)
    sep_arr = np.array([r["always_separated_cpwer"] for r in rows], dtype=float)
    rv2_arr = np.array([r["router_v2_cpwer"] for r in rows], dtype=float)
    oracle_arr = np.array([r["oracle_best_cpwer"] for r in rows], dtype=float)

    kl_point = float(kl_arr.mean())
    lang_point = float(lang_arr.mean())
    or_point = float(or_arr.mean())
    and_point = float(and_arr.mean())
    mixed_point = float(mixed_arr.mean())
    sep_point = float(sep_arr.mean())
    rv2_point = float(rv2_arr.mean())
    oracle_point = float(oracle_arr.mean())

    # --- decision counts
    def _counts(key: str) -> dict[str, int]:
        return {
            "mixed": sum(1 for r in rows if r[key] == "mixed"),
            "separated": sum(1 for r in rows if r[key] == "separated"),
        }

    kl_counts = _counts("kl_decision")
    lang_counts = _counts("lang_id_decision")
    or_counts = _counts("or_decision")
    and_counts = _counts("and_decision")

    # --- detector evaluation for OR and AND ensembles
    halluc_label_ints = [1 if l["hallucinated"] else 0 for l in labels]
    mode_s_mask = [l["mode_s"] for l in labels]
    diverse_mask = [l["diverse_hallucination"] for l in labels]

    or_flags_ints = [r["or_flag"] for r in rows]
    and_flags_ints = [r["and_flag"] for r in rows]

    # OR ensemble confusion-matrix metrics
    or_tp = sum(1 for r, l in zip(rows, labels) if l["hallucinated"] and r["or_flag"])
    or_fp = sum(1 for r, l in zip(rows, labels) if not l["hallucinated"] and r["or_flag"])
    or_tn = sum(1 for r, l in zip(rows, labels) if not l["hallucinated"] and not r["or_flag"])
    or_fn = sum(1 for r, l in zip(rows, labels) if l["hallucinated"] and not r["or_flag"])
    or_sens = or_tp / n_halluc if n_halluc else 0.0
    or_spec = or_tn / n_nonhalluc if n_nonhalluc else 1.0
    or_ms_tp = sum(1 for r, l in zip(rows, labels) if l["mode_s"] and r["or_flag"])
    or_div_tp = sum(1 for r, l in zip(rows, labels) if l["diverse_hallucination"] and r["or_flag"])

    # AND ensemble confusion-matrix metrics
    and_tp = sum(1 for r, l in zip(rows, labels) if l["hallucinated"] and r["and_flag"])
    and_fp = sum(1 for r, l in zip(rows, labels) if not l["hallucinated"] and r["and_flag"])
    and_tn = sum(1 for r, l in zip(rows, labels) if not l["hallucinated"] and not r["and_flag"])
    and_fn = sum(1 for r, l in zip(rows, labels) if l["hallucinated"] and not r["and_flag"])
    and_sens = and_tp / n_halluc if n_halluc else 0.0
    and_spec = and_tn / n_nonhalluc if n_nonhalluc else 1.0
    and_ms_tp = sum(1 for r, l in zip(rows, labels) if l["mode_s"] and r["and_flag"])
    and_div_tp = sum(1 for r, l in zip(rows, labels) if l["diverse_hallucination"] and r["and_flag"])
    and_fp_rate = and_fp / n_nonhalluc if n_nonhalluc else 0.0

    # --- KL-alone detector metrics (for comparison)
    kl_tp = sum(1 for r, l in zip(rows, labels) if l["hallucinated"] and r["kl_flag"])
    kl_fp = sum(1 for r, l in zip(rows, labels) if not l["hallucinated"] and r["kl_flag"])
    kl_ms_tp = sum(1 for r, l in zip(rows, labels) if l["mode_s"] and r["kl_flag"])

    # --- lang-id-alone detector metrics (for comparison)
    lang_tp = sum(1 for r, l in zip(rows, labels) if l["hallucinated"] and r["lang_id_flag"])
    lang_fp = sum(1 for r, l in zip(rows, labels) if not l["hallucinated"] and r["lang_id_flag"])
    lang_ms_tp = sum(1 for r, l in zip(rows, labels) if l["mode_s"] and r["lang_id_flag"])

    # --- bootstrap CIs (B=10000, seed=42, RQ39 framework)
    or_boot = bootstrap_distribution(or_arr, N_BOOT, SEED)
    or_pct_ci = percentile_ci(or_boot)
    or_bca_ci = bca_ci(or_arr, or_boot)

    and_boot = bootstrap_distribution(and_arr, N_BOOT, SEED)
    and_pct_ci = percentile_ci(and_boot)
    and_bca_ci = bca_ci(and_arr, and_boot)

    # paired deltas: OR vs KL-alone, OR vs lang-id, OR vs always-mixed
    or_minus_kl = paired_delta_distribution(or_arr, kl_arr, N_BOOT, SEED)
    or_minus_lang = paired_delta_distribution(or_arr, lang_arr, N_BOOT, SEED)
    or_minus_mixed = paired_delta_distribution(or_arr, mixed_arr, N_BOOT, SEED)
    or_minus_kl_ci = percentile_ci(or_minus_kl)
    or_minus_lang_ci = percentile_ci(or_minus_lang)
    or_minus_mixed_ci = percentile_ci(or_minus_mixed)

    # paired deltas: AND vs KL-alone, AND vs lang-id
    and_minus_kl = paired_delta_distribution(and_arr, kl_arr, N_BOOT, SEED)
    and_minus_lang = paired_delta_distribution(and_arr, lang_arr, N_BOOT, SEED)
    and_minus_kl_ci = percentile_ci(and_minus_kl)
    and_minus_lang_ci = percentile_ci(and_minus_lang)

    # --- regret analysis
    regret_or_vs_oracle = or_point - oracle_point
    regret_and_vs_oracle = and_point - oracle_point
    regret_kl_vs_oracle = kl_point - oracle_point
    regret_lang_vs_oracle = lang_point - oracle_point
    regret_mixed_vs_oracle = mixed_point - oracle_point

    # --- Mode S per-window detail
    mode_s_rows = [r for r in rows if r["mode_s"]]
    mode_s_detail = {
        str(r["window_id"]): {
            "kl_score": r["kl_score"],
            "lang_id_entropy": r["lang_id_entropy"],
            "kl_flag": r["kl_flag"],
            "lang_id_flag": r["lang_id_flag"],
            "or_flag": r["or_flag"],
            "and_flag": r["and_flag"],
            "kl_decision": r["kl_decision"],
            "lang_id_decision": r["lang_id_decision"],
            "or_decision": r["or_decision"],
            "and_decision": r["and_decision"],
            "or_cpwer": r["or_cpwer"],
            "and_cpwer": r["and_cpwer"],
            "always_mixed_cpwer": r["always_mixed_cpwer"],
            "always_separated_cpwer": r["always_separated_cpwer"],
        }
        for r in mode_s_rows
    }

    # --- hypothesis verdicts
    # H60a: OR ensemble cpWER < 1.030 (beats KL-alone). KILLED if >= 1.030.
    h60a_supported = or_point < H60A_KILL_THRESHOLD
    h60a_killed = or_point >= H60A_KILL_THRESHOLD

    # H60b: OR catches 100% Mode S AND >= 90% all halluc. KILLED if < 100% Mode S OR < 90% all.
    or_mode_s_sens = or_ms_tp / n_mode_s if n_mode_s else 0.0
    or_all_sens = or_sens
    h60b_supported = (
        or_mode_s_sens >= H60B_MODE_S_SENS_KILL
        and or_all_sens >= H60B_ALL_SENS_KILL
    )

    # H60c: AND FP rate < 7.5% (3/40). KILLED if AND FP >= 7.5%.
    h60c_supported = and_fp_rate < AND_FLAG_FP_KILL_RATE

    summary: dict[str, Any] = {
        "label": "experimental/frontier",
        "rq": "RQ60: KL+lang-id ensemble corrected router",
        "closes_issue": 984,
        "source_data": str(SRC_JSON.relative_to(PROJECT_ROOT)),
        "source_label": "external/sanity-check",
        "method": (
            "Reanalysis only (no Whisper / no ASR / no LLM run). Tests OR and AND "
            "ensembles of two complementary detectors: (1) KL detector (RQ58, "
            "character 2-gram KL-divergence, threshold 5.418144, catches Mode S "
            "but has 4 FPs, cpWER 1.030303) and (2) lang-id entropy (RQ13/RQ16, "
            "Shannon entropy over Unicode script categories, threshold 0.38, "
            "catches 94.6% of diverse hallucinations but 0% of Mode S, cpWER "
            "1.04329). OR ensemble: MIXED if KL >= 5.418144 OR lang_id >= 0.38 "
            "(superset of each detector's flags). AND ensemble: MIXED if KL >= "
            "5.418144 AND lang_id >= 0.38 (intersection, fewer FPs). Per-window "
            "cpWER is the chosen route's stored word-level cpWER (matches RQ16/"
            "RQ58 bit-for-bit). Bootstrap 10,000 resamples, seed=42, percentile + "
            "BCa CIs (RQ39 framework). NOTE: lang-id threshold 0.38 gives "
            "IDENTICAL results to RQ16's 0.409 (no score falls between them)."
        ),
        "meeting_id": data["meeting_id"],
        "n_windows": n,
        "n_hallucinated": n_halluc,
        "n_nonhallucinated": n_nonhalluc,
        "n_mode_s": n_mode_s,
        "n_diverse_hallucination": n_diverse,
        "mode_s_window_ids": mode_s_ids,
        "hallucination_label": "always_separated_cpwer > 1.0 (37/40 split, RQ12)",
        "mode_s_definition": (
            "hallucinated AND lang_id_entropy < 0.409 AND length_ratio < 2.0 "
            "AND cr < 2.4 (RQ19 verified windows 22, 30)"
        ),
        "thresholds": {
            "kl_threshold": KL_THRESHOLD,
            "kl_threshold_note": (
                "RQ58's empirically-calibrated threshold at >=90% specificity "
                "(PR #981). Task spec says '5.42' (rounded); we use the precise "
                "value 5.418144 for exact reproducibility."
            ),
            "lang_id_threshold": LANG_ID_THRESHOLD,
            "lang_id_threshold_note": (
                "Task spec pre-registers 0.38. Gives IDENTICAL results to RQ16's "
                "0.409 (no lang-id score falls between 0.38 and 0.409); both "
                "reproduce RQ16's 1.043290 with 3/40 FPs, 94.6% sensitivity."
            ),
        },
        "ngram_config": {
            "n": N_GRAM,
            "reference": (
                "average 2-gram distribution of 40 non-hallucinated tracks' "
                "concatenated separated text (RQ34 separated_concat)"
            ),
            "vocab_size": ref_vocab_size,
            "metric": "KL(text || reference), additive smoothing 1e-9",
            "aggregation": "MAX across per-speaker separated tracks (RQ12/RQ13 worst-case)",
        },
        "baselines": {
            "always_mixed_cpwer": _round6(mixed_point),
            "always_separated_cpwer": _round6(sep_point),
            "router_v2_cpwer": _round6(rv2_point),
            "oracle_best_cpwer": _round6(oracle_point),
        },
        "sanity_checks": {
            "kl_alone_cpwer_recomputed": _round6(kl_alone_cpwer),
            "kl_alone_cpwer_reference": RQ58_KL_CPWER,
            "kl_alone_matches_rq58": bool(abs(kl_alone_cpwer - RQ58_KL_CPWER) < 1e-4),
            "lang_id_alone_cpwer_recomputed": _round6(lang_id_alone_cpwer),
            "lang_id_alone_cpwer_reference": RQ16_LANG_ID_CPWER,
            "lang_id_alone_matches_rq16": bool(abs(lang_id_alone_cpwer - RQ16_LANG_ID_CPWER) < 1e-4),
            "note": (
                "KL-alone recomputed with threshold 5.418144 must match RQ58's "
                "1.030303; lang-id-alone recomputed with threshold 0.38 must match "
                "RQ16's 1.043290. Both are bit-for-bit reproductions."
            ),
        },
        "ensemble_results": {
            "or_ensemble": {
                "cpwer": _round6(or_point),
                "percentile_ci_95": _ci_pair(or_pct_ci),
                "bca_ci_95": _ci_pair(or_bca_ci),
                "decision_counts": or_counts,
                "n_flagged": int(sum(or_flags_ints)),
                "detector_eval": {
                    "sensitivity_all": _round6(or_sens),
                    "specificity": _round6(or_spec),
                    "sensitivity_mode_s": _round6(or_mode_s_sens),
                    "tp": or_tp, "fp": or_fp, "tn": or_tn, "fn": or_fn,
                    "tp_mode_s": or_ms_tp, "n_mode_s": n_mode_s,
                    "tp_diverse": or_div_tp, "n_diverse": n_diverse,
                },
            },
            "and_ensemble": {
                "cpwer": _round6(and_point),
                "percentile_ci_95": _ci_pair(and_pct_ci),
                "bca_ci_95": _ci_pair(and_bca_ci),
                "decision_counts": and_counts,
                "n_flagged": int(sum(and_flags_ints)),
                "detector_eval": {
                    "sensitivity_all": _round6(and_sens),
                    "specificity": _round6(and_spec),
                    "sensitivity_mode_s": _round6(and_ms_tp / n_mode_s if n_mode_s else 0.0),
                    "fp_rate": _round6(and_fp_rate),
                    "tp": and_tp, "fp": and_fp, "tn": and_tn, "fn": and_fn,
                    "tp_mode_s": and_ms_tp, "n_mode_s": n_mode_s,
                    "tp_diverse": and_div_tp, "n_diverse": n_diverse,
                },
            },
        },
        "individual_detector_context": {
            "kl_alone": {
                "cpwer": _round6(kl_point),
                "reference_cpwer": RQ58_KL_CPWER,
                "threshold": KL_THRESHOLD,
                "tp": kl_tp, "fp": kl_fp,
                "tp_mode_s": kl_ms_tp, "n_mode_s": n_mode_s,
                "decision_counts": kl_counts,
            },
            "lang_id_alone": {
                "cpwer": _round6(lang_point),
                "reference_cpwer": RQ16_LANG_ID_CPWER,
                "threshold": LANG_ID_THRESHOLD,
                "tp": lang_tp, "fp": lang_fp,
                "tp_mode_s": lang_ms_tp, "n_mode_s": n_mode_s,
                "decision_counts": lang_counts,
            },
        },
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "alpha": ALPHA,
            "convention": "rng.integers(0, n, size=n) per resample (RQ16/RQ39 verbatim)",
        },
        "regret_analysis": {
            "or_regret_vs_oracle": _round6(regret_or_vs_oracle),
            "and_regret_vs_oracle": _round6(regret_and_vs_oracle),
            "kl_regret_vs_oracle": _round6(regret_kl_vs_oracle),
            "lang_id_regret_vs_oracle": _round6(regret_lang_vs_oracle),
            "always_mixed_regret_vs_oracle": _round6(regret_mixed_vs_oracle),
        },
        "paired_delta_cis": {
            "or_minus_kl_point": _round6(or_point - kl_point),
            "or_minus_kl_ci_95": _ci_pair(or_minus_kl_ci),
            "or_minus_lang_id_point": _round6(or_point - lang_point),
            "or_minus_lang_id_ci_95": _ci_pair(or_minus_lang_ci),
            "or_minus_always_mixed_point": _round6(or_point - mixed_point),
            "or_minus_always_mixed_ci_95": _ci_pair(or_minus_mixed_ci),
            "and_minus_kl_point": _round6(and_point - kl_point),
            "and_minus_kl_ci_95": _ci_pair(and_minus_kl_ci),
            "and_minus_lang_id_point": _round6(and_point - lang_point),
            "and_minus_lang_id_ci_95": _ci_pair(and_minus_lang_ci),
        },
        "mode_s_detail": mode_s_detail,
        "hypothesis_verdicts": {
            "H60a": {
                "statement": (
                    "OR ensemble cpWER < 1.030 (beats KL-alone, the current best). "
                    "KILLED if cpWER >= 1.030."
                ),
                "or_cpwer": _round6(or_point),
                "kl_alone_cpwer": RQ58_KL_CPWER,
                "kill_threshold": H60A_KILL_THRESHOLD,
                "delta_or_minus_kl": _round6(or_point - kl_point),
                "paired_delta_ci_95": _ci_pair(or_minus_kl_ci),
                "killed": bool(h60a_killed),
                "supported": bool(h60a_supported),
                "reason": (
                    f"OR ensemble cpWER = {or_point:.6f} "
                    f"{'<' if h60a_supported else '>='} kill threshold "
                    f"{H60A_KILL_THRESHOLD:.3f}. "
                    f"{'OR beats KL-alone.' if h60a_supported else 'OR does NOT beat KL-alone.'} "
                    f"(KL-alone precise: {RQ58_KL_CPWER:.6f}, delta = {or_point-kl_point:+.6f})"
                ),
            },
            "H60b": {
                "statement": (
                    "OR ensemble catches 100% Mode S (from KL) AND >= 90% all "
                    "hallucinations. KILLED if < 100% Mode S OR < 90% all."
                ),
                "or_mode_s_sensitivity": _round6(or_mode_s_sens),
                "or_all_sensitivity": _round6(or_all_sens),
                "tp_mode_s": or_ms_tp,
                "n_mode_s": n_mode_s,
                "tp_all": or_tp,
                "n_all": n_halluc,
                "mode_s_window_ids": mode_s_ids,
                "supported": bool(h60b_supported),
                "reason": (
                    f"OR Mode S sensitivity = {or_mode_s_sens:.0%} "
                    f"({or_ms_tp}/{n_mode_s}); all-hallucination sensitivity = "
                    f"{or_all_sens:.1%} ({or_tp}/{n_halluc}). "
                    f"{'OR catches all Mode S and >= 90% all.' if h60b_supported else 'OR fails the sensitivity criterion.'}"
                ),
            },
            "H60c": {
                "statement": (
                    "AND ensemble FP rate < 7.5% (3/40). KILLED if AND FP >= 7.5%."
                ),
                "and_fp_rate": _round6(and_fp_rate),
                "and_fp": and_fp,
                "n_nonhallucinated": n_nonhalluc,
                "kill_rate": AND_FLAG_FP_KILL_RATE,
                "kl_alone_fp": kl_fp,
                "lang_id_alone_fp": lang_fp,
                "supported": bool(h60c_supported),
                "reason": (
                    f"AND FP rate = {and_fp_rate:.1%} ({and_fp}/{n_nonhalluc}) "
                    f"{'<' if h60c_supported else '>='} kill rate "
                    f"{AND_FLAG_FP_KILL_RATE:.1%} (3/40). "
                    f"KL-alone has {kl_fp} FPs; lang-id-alone has {lang_fp} FPs."
                ),
            },
        },
        "per_window": rows,
    }

    # ----------------------------------------------------------- write CSV
    csv_fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # ----------------------------------------------------------- write JSON
    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------- console
    print(f"=== RQ60: KL+lang-id ensemble corrected router ({n} windows) ===")
    print(f"Label: experimental/frontier  |  Source: {SRC_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Hallucinated: {n_halluc}  |  non-hallucinated: {n_nonhalluc}  "
          f"|  Mode S: {n_mode_s} ({mode_s_ids})  |  diverse: {n_diverse}")
    print()
    print(f"Thresholds: KL = {KL_THRESHOLD} (RQ58), lang-id = {LANG_ID_THRESHOLD} (task spec)")
    print(f"  NOTE: lang-id 0.38 gives IDENTICAL results to RQ16's 0.409.")
    print()
    print("Sanity checks (must reproduce RQ58 and RQ16):")
    print(f"  KL-alone cpwer      : {kl_alone_cpwer:.6f}  (RQ58: {RQ58_KL_CPWER})  "
          f"{'OK' if abs(kl_alone_cpwer - RQ58_KL_CPWER) < 1e-4 else 'MISMATCH'}")
    print(f"  lang-id-alone cpwer : {lang_id_alone_cpwer:.6f}  (RQ16: {RQ16_LANG_ID_CPWER})  "
          f"{'OK' if abs(lang_id_alone_cpwer - RQ16_LANG_ID_CPWER) < 1e-4 else 'MISMATCH'}")
    print()
    print("Detector evaluation:")
    print(f"  {'detector':20s} {'sens_all':>9s} {'sens_ms':>8s} {'spec':>6s} {'fp':>4s} {'tp':>4s}")
    print(f"  {'KL-alone':20s} {kl_tp/n_halluc:9.1%} {kl_ms_tp/n_mode_s:8.0%} "
          f"{1-kl_fp/n_nonhalluc:6.1%} {kl_fp:4d} {kl_tp:4d}")
    print(f"  {'lang-id-alone':20s} {lang_tp/n_halluc:9.1%} {lang_ms_tp/n_mode_s:8.0%} "
          f"{1-lang_fp/n_nonhalluc:6.1%} {lang_fp:4d} {lang_tp:4d}")
    print(f"  {'OR ensemble':20s} {or_sens:9.1%} {or_mode_s_sens:8.0%} "
          f"{or_spec:6.1%} {or_fp:4d} {or_tp:4d}")
    print(f"  {'AND ensemble':20s} {and_sens:9.1%} {and_ms_tp/n_mode_s:8.0%} "
          f"{and_spec:6.1%} {and_fp:4d} {and_tp:4d}")
    print()
    print("Aggregate cpWER (mean over 77 windows):")
    print(f"  always_mixed     : {mixed_point:.6f}")
    print(f"  always_separated : {sep_point:.6f}")
    print(f"  router_v2        : {rv2_point:.6f}")
    print(f"  oracle_best      : {oracle_point:.6f}")
    print(f"  KL-alone (RQ58)  : {kl_point:.6f}")
    print(f"  lang-id (RQ16)   : {lang_point:.6f}")
    print(f"  OR ensemble      : {or_point:.6f}  "
          f"pct CI [{or_pct_ci[0]:.6f}, {or_pct_ci[1]:.6f}]  "
          f"BCa CI [{or_bca_ci[0]:.6f}, {or_bca_ci[1]:.6f}]")
    print(f"  AND ensemble     : {and_point:.6f}  "
          f"pct CI [{and_pct_ci[0]:.6f}, {and_pct_ci[1]:.6f}]  "
          f"BCa CI [{and_bca_ci[0]:.6f}, {and_bca_ci[1]:.6f}]")
    print()
    print(f"Decision counts: KL mixed={kl_counts['mixed']}, "
          f"lang-id mixed={lang_counts['mixed']}, "
          f"OR mixed={or_counts['mixed']}, AND mixed={and_counts['mixed']}")
    print()
    print("Paired-delta CIs:")
    print(f"  OR - KL       : {or_point-kl_point:+.6f}  CI [{or_minus_kl_ci[0]:+.6f}, {or_minus_kl_ci[1]:+.6f}]")
    print(f"  OR - lang-id  : {or_point-lang_point:+.6f}  CI [{or_minus_lang_ci[0]:+.6f}, {or_minus_lang_ci[1]:+.6f}]")
    print(f"  OR - mixed    : {or_point-mixed_point:+.6f}  CI [{or_minus_mixed_ci[0]:+.6f}, {or_minus_mixed_ci[1]:+.6f}]")
    print(f"  AND - KL      : {and_point-kl_point:+.6f}  CI [{and_minus_kl_ci[0]:+.6f}, {and_minus_kl_ci[1]:+.6f}]")
    print(f"  AND - lang-id : {and_point-lang_point:+.6f}  CI [{and_minus_lang_ci[0]:+.6f}, {and_minus_lang_ci[1]:+.6f}]")
    print()
    print("Mode S per-window detail:")
    for wid, detail in mode_s_detail.items():
        print(f"  window {wid}: KL={detail['kl_score']} (flag={detail['kl_flag']}), "
              f"lang_id={detail['lang_id_entropy']} (flag={detail['lang_id_flag']}), "
              f"OR={detail['or_decision']}, AND={detail['and_decision']}")
    print()
    print("Hypothesis verdicts:")
    print(f"  H60a (OR cpWER < 1.030, beats KL-alone): "
          f"{'SUPPORTED' if h60a_supported else 'KILLED'}  "
          f"(OR={or_point:.6f}, KL={RQ58_KL_CPWER:.6f}, "
          f"delta={or_point-kl_point:+.6f}, kill_thresh={H60A_KILL_THRESHOLD})")
    print(f"  H60b (OR 100% Mode S AND >= 90% all): "
          f"{'SUPPORTED' if h60b_supported else 'KILLED'}  "
          f"(Mode S sens={or_mode_s_sens:.0%} ({or_ms_tp}/{n_mode_s}), "
          f"all sens={or_all_sens:.1%} ({or_tp}/{n_halluc}))")
    print(f"  H60c (AND FP rate < 7.5%): "
          f"{'SUPPORTED' if h60c_supported else 'KILLED'}  "
          f"(AND FP={and_fp}/{n_nonhalluc} = {and_fp_rate:.1%}, "
          f"kill_rate={AND_FLAG_FP_KILL_RATE:.1%})")
    print()
    print(f"Wrote: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
