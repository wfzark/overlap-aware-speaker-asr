"""RQ49: Speaker-count stratified threshold for reduced modality.

REANALYSIS ONLY -- no Whisper / no ASR model is run. Reads the existing
AISHELL-4 external-validation results (``results/external_sanity_check/aishell4/
rq1_aishell4_validation_results.json``, label ``external/sanity-check``, PR #890)
and tests whether stratifying the lang-id entropy threshold by active-speaker
count reduces the 6-modality that RQ44 (PR #963) observed in the pooled
bootstrap threshold distribution.

Motivation (RQ38, PR #948 / RQ44, PR #963)
------------------------------------------
RQ38 showed the separated-track hallucination rate is monotone in the number of
ACTIVE speakers (Spearman rho = +0.611; 0% -> 34.5% -> 65% -> 93.3% across
0 -> 1 -> 2 -> 3+ active speakers). RQ44 showed the pooled bootstrap threshold
distribution is 6-modal (6 unique thresholds, 5 with >= 5% frequency; width 0.94;
median OOB cpWER 1.056). RQ44's FINDINGS note the modes are partly driven by the
2 Mode S (low-entropy hallucinated) windows -- w22 and w30 -- which RQ38's H38b
confirmed occur only with <= 2 active speakers. RQ49 asks: does stratifying the
threshold by active-speaker count (<= 2 vs > 2) reduce the modality?

Method (RQ49)
-------------
1. Active speakers = count of non-empty ``separated_text_per_speaker`` values
   (RQ38's definition, verbatim). This is the speaker-count proxy RQ38 showed
   carries the hallucination effect (rho = +0.611).
2. Stratify the 77 windows into:
      Stratum 1 (<= 2 active speakers) -- this is where Mode S occurs (w22, w30)
      Stratum 2 (>  2 active speakers)
3. For each stratum: B = 2000 bootstrap resamples (seed = 42, n = stratum_size
   with replacement). On each resample: calibrate the lang-id entropy threshold
   maximising sensitivity at >= 90% specificity (RQ13/RQ16/RQ25/RQ44 rule,
   verbatim) on the in-bag windows; evaluate the corrected-router cpWER on the
   out-of-bag windows.
4. Stratified router: on each (aligned) bootstrap resample, calibrate a
   threshold for BOTH strata, then evaluate the COMBINED OOB cpWER -- each OOB
   window is routed by its own stratum's threshold. The combined OOB pools the
   OOB windows from both strata.

Routing rule (RQ13/RQ16/RQ25/RQ44 convention)
---------------------------------------------
HIGH lang-id entropy = diverse multilingual gibberish = hallucination. The
detector flags the separated track when ``lang_id_entropy >= threshold``:

    if lang_id_entropy >= threshold -> route MIXED  (cpWER = always_mixed_cpwer)
    else                            -> route SEPARATED (cpWER = always_separated_cpwer)

Pre-registered hypotheses (issue for RQ49)
------------------------------------------
- H49a: <= 2-speaker threshold distribution has <= 3 modes (>= 5% frequency).
        Kill: > 3 modes with >= 5% frequency.  (RQ44 pooled: 5 modes >= 5%.)
- H49b: >  2-speaker threshold distribution has <= 2 modes (>= 5% frequency).
        Kill: > 2 modes with >= 5% frequency.
- H49c: stratified router median OOB cpWER < RQ44's 1.056.
        Kill: >= 1.056.

This script is pure reanalysis (numpy + stdlib only; scipy / sklearn / Whisper
are NOT required). The detector primitives (``script_category``,
``language_id_entropy``, ``max_across_speakers``) and the calibration rule
(``calibrate_threshold_at_spec``) are lifted verbatim from RQ44 so thresholds
are directly comparable.

Label: experimental/frontier. Builds on RQ13 (PR #904), RQ16 (PR #912),
RQ25 (PR #929), RQ38 (PR #948), and RQ44 (PR #963).
"""
from __future__ import annotations

import csv
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_JSON = (
    PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)
OUT_DIR = PROJECT_ROOT / "results" / "frontier" / "stratified_threshold"
OUT_CSV = OUT_DIR / "stratified_threshold_results.csv"
OUT_JSON = OUT_DIR / "stratified_threshold_results.json"

# ------------------------------------------------------------------ constants
CATASTROPHIC_CPWER = 1.0    # cpWER > 1.0 => insertions dominate (hallucination label)
TARGET_SPECIFICITY = 0.90   # calibrate the threshold to >= 90% specificity
THRESHOLD_GRID = [round(0.01 * i, 2) for i in range(0, 201)]  # 0.00, 0.01, ..., 2.00
N_BOOT = 2000               # per stratum (kept modest for runtime; RQ44 used 10000)
SEED = 42
EPS = 1e-9
SPEAKER_SPLIT = 2           # stratum 1: <= SPEAKER_SPLIT active speakers
MIN_MODE_FRACTION = 0.05    # a "mode" is a threshold value with >= 5% frequency

# RQ44 published references (B=10000, seed=42, pooled over 77 windows).
RQ44_POOLED_N_UNIQUE = 6
RQ44_POOLED_N_MODES_5PCT = 5
RQ44_POOLED_WIDTH = 0.94
RQ44_POOLED_MEDIAN_OOB_CPWER = 1.055556   # exact value stored in RQ44's JSON (19/18)
RQ44_POOLED_MEDIAN_OOB_CPWER_ROUNDED = 1.056  # the rounded figure cited in the RQ49 spec
RQ44_IN_SAMPLE_THRESHOLD = 0.38  # RQ25's in-sample threshold on the 77 windows

# Hypothesis kill thresholds. H49c's kill is the rounded RQ44 figure ">= 1.056"
# as stated in the RQ49 spec; the exact RQ44 pooled median (1.055556) is kept
# separately for the substantive tie comparison.
H49A_MAX_MODES = 3           # <= 2-speaker stratum: kill if > 3 modes (>= 5%)
H49B_MAX_MODES = 2           # >  2-speaker stratum: kill if > 2 modes (>= 5%)
H49C_MAX_CPWER = RQ44_POOLED_MEDIAN_OOB_CPWER_ROUNDED  # stratified router: kill if >= 1.056


# ------------------------------------------------------------- script detection
def script_category(ch: str) -> str:
    """Map a character to a coarse Unicode script category (RQ13/RQ16/RQ25 verbatim).

    Uses ``unicodedata.name``. Whitespace -> "Space"; punctuation/symbols ->
    "Punct"; control/unknown -> "Other". Sufficient to separate Han / Latin /
    Hiragana / Katakana / Hangul / Cyrillic / Arabic / Greek / Digit, which are
    exactly the scripts RQ12/RQ13 observed in AISHELL-4 hallucination."""
    if ch.isspace():
        return "Space"
    name = unicodedata.name(ch, "")
    if not name:
        return "Other"
    first = name.split()[0]
    if first == "CJK":
        return "Han"
    if first == "LATIN" or "LATIN" in name:
        return "Latin"
    if first == "HIRAGANA":
        return "Hiragana"
    if first == "KATAKANA":
        return "Katakana"
    if first == "HANGUL":
        return "Hangul"
    if first == "CYRILLIC":
        return "Cyrillic"
    if first == "ARABIC":
        return "Arabic"
    if first == "GREEK":
        return "Greek"
    if first == "DIGIT":
        return "Digit"
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "Punct"
    return "Other"


# --------------------------------------------------------------- the detector
def language_id_entropy(text: str) -> float:
    """Shannon entropy (bits) over the script-category distribution of the text (RQ13).

    Clean Chinese (near-monoscript Han) -> entropy ~ 0. Diverse multilingual
    gibberish mixing Han+Latin+Katakana+Hangul -> high entropy."""
    if not text or not text.strip():
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        sc = script_category(ch)
        counts[sc] = counts.get(sc, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def max_across_speakers(window: dict[str, Any]) -> float:
    """Max of ``language_id_entropy`` over the per-speaker separated transcripts
    (worst-case speaker track; RQ12/RQ13 convention). Empty/whitespace speaker
    texts are effectively skipped."""
    vals = [
        language_id_entropy(str(t))
        for t in window.get("separated_text_per_speaker", {}).values()
        if t is not None and str(t).strip()
    ]
    return max(vals) if vals else 0.0


def count_active_speakers(separated_text_per_speaker: dict[str, Any]) -> int:
    """Count of non-empty ``separated_text_per_speaker`` values (RQ38 verbatim).

    A speaker is "active" if Whisper produced a non-empty transcript for that
    speaker track. This is the speaker-count proxy RQ38 showed carries the
    hallucination effect (Spearman rho = +0.611)."""
    return sum(
        1 for v in separated_text_per_speaker.values()
        if v is not None and str(v).strip()
    )


# --------------------------------------------------------------- pure helpers
def stratify_by_speaker_count(
    active_speakers: np.ndarray,
    split: int = SPEAKER_SPLIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition window indices by active-speaker count.

    Returns ``(idx_le, idx_gt)`` where ``idx_le`` are the indices with
    ``active_speakers <= split`` (stratum 1) and ``idx_gt`` are the indices with
    ``active_speakers > split`` (stratum 2). Pure: deterministic, no I/O."""
    arr = np.asarray(active_speakers, dtype=int)
    idx_le = np.where(arr <= split)[0]
    idx_gt = np.where(arr > split)[0]
    return idx_le, idx_gt


def calibrate_threshold_at_spec(
    scores: np.ndarray,
    labels: np.ndarray,
    grid: list[float] | None = None,
    target_spec: float = TARGET_SPECIFICITY,
) -> dict[str, Any]:
    """Sweep threshold over ``grid`` (default THRESHOLD_GRID); select the threshold
    with specificity >= ``target_spec`` and maximal sensitivity. Tie-breaker:
    higher specificity, then lower threshold (more sensitive).

    Convention (RQ13/RQ16/RQ25/RQ44): ``score >= threshold`` flags the window as
    hallucinated. Sensitivity = TP / (TP + FN); specificity = TN / (TN + FP).

    Returns the chosen threshold plus its sensitivity, specificity, and
    confusion counts. Verbatim copy of RQ44's ``calibrate_threshold_at_spec``
    so thresholds are directly comparable across RQ44 and RQ49.
    """
    if grid is None:
        grid = THRESHOLD_GRID
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = int(len(pos))
    n_neg = int(len(neg))
    best: dict[str, Any] | None = None
    for t in grid:
        fp = int(np.sum(neg >= t - EPS))
        tp = int(np.sum(pos >= t - EPS))
        tn = n_neg - fp
        fn = n_pos - tp
        spec = (tn / n_neg) if n_neg > 0 else 1.0
        sens = (tp / n_pos) if n_pos > 0 else 0.0
        if spec >= target_spec - EPS:
            cand = {
                "threshold": float(t),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            if best is None:
                best = cand
            else:
                # Maximise sensitivity; tie-break by higher specificity, then
                # lower threshold (more sensitive, fewer false negatives).
                if sens > best["sensitivity"] + EPS:
                    best = cand
                elif abs(sens - best["sensitivity"]) <= EPS:
                    if spec > best["specificity"] + EPS:
                        best = cand
                    elif abs(spec - best["specificity"]) <= EPS and t < best["threshold"]:
                        best = cand
    if best is None:
        # No threshold satisfies the specificity target -> fall back to the
        # highest threshold (most conservative: flag nothing).
        t_max = float(grid[-1]) if grid else 1.0
        best = {
            "threshold": t_max, "sensitivity": 0.0, "specificity": 1.0,
            "tp": 0, "fp": 0, "tn": n_neg, "fn": n_pos,
        }
    return best


def percentile_interval(
    values: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    """Return the ``(lo, hi)`` percentile interval of ``values`` using numpy's
    default linear interpolation. Returns ``(nan, nan)`` for empty input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def count_modes(thresholds: np.ndarray, min_fraction: float = MIN_MODE_FRACTION) -> dict[str, Any]:
    """Summarise a bootstrap threshold distribution's modality.

    A "mode" is a distinct threshold value whose frequency is >= ``min_fraction``
    (default 5%). Returns the number of modes, the total number of distinct
    values, and the mode table (value, count, fraction) sorted by descending
    count. Pure: deterministic, no I/O."""
    arr = np.asarray(thresholds, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n_modes": 0, "n_unique": 0, "modes": []}
    uniq, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)  # descending by count
    uniq = uniq[order]
    counts = counts[order]
    modes = [
        {"threshold": float(u), "count": int(c), "fraction": float(c / n)}
        for u, c in zip(uniq, counts)
        if (c / n) >= min_fraction - EPS
    ]
    return {"n_modes": len(modes), "n_unique": int(uniq.size), "modes": modes}


def combined_oob_cpwer(
    oob_scores_1: np.ndarray,
    oob_mixed_1: np.ndarray,
    oob_sep_1: np.ndarray,
    thr_1: float,
    oob_scores_2: np.ndarray,
    oob_mixed_2: np.ndarray,
    oob_sep_2: np.ndarray,
    thr_2: float,
) -> dict[str, Any]:
    """Combined out-of-bag cpWER for the stratified router.

    Stratum-1 OOB windows are routed by ``thr_1``; stratum-2 OOB windows are
    routed by ``thr_2`` (each window uses its OWN stratum's threshold). Routing:
    ``score >= threshold`` -> MIXED (``oob_mixed``); else SEPARATED (``oob_sep``).
    The combined cpWER is the mean selected cpWER over the union of both strata's
    OOB windows. Returns ``cpwer = nan`` if both OOB sets are empty. Pure."""
    s1 = np.asarray(oob_scores_1, dtype=float)
    m1 = np.asarray(oob_mixed_1, dtype=float)
    p1 = np.asarray(oob_sep_1, dtype=float)
    s2 = np.asarray(oob_scores_2, dtype=float)
    m2 = np.asarray(oob_mixed_2, dtype=float)
    p2 = np.asarray(oob_sep_2, dtype=float)
    if s1.size > 0:
        flag1 = s1 >= thr_1 - EPS
        sel1 = np.where(flag1, m1, p1)
        n_flag1 = int(flag1.sum())
        n_sep1 = int((~flag1).sum())
    else:
        sel1 = np.array([], dtype=float)
        n_flag1 = 0
        n_sep1 = 0
    if s2.size > 0:
        flag2 = s2 >= thr_2 - EPS
        sel2 = np.where(flag2, m2, p2)
        n_flag2 = int(flag2.sum())
        n_sep2 = int((~flag2).sum())
    else:
        sel2 = np.array([], dtype=float)
        n_flag2 = 0
        n_sep2 = 0
    all_sel = np.concatenate([sel1, sel2]) if sel1.size + sel2.size > 0 else np.array([], dtype=float)
    if all_sel.size == 0:
        return {"cpwer": float("nan"), "n_oob": 0,
                "n_oob_s1": int(s1.size), "n_oob_s2": int(s2.size),
                "n_flagged_mixed": 0, "n_separated": 0}
    return {
        "cpwer": float(all_sel.mean()),
        "n_oob": int(all_sel.size),
        "n_oob_s1": int(s1.size),
        "n_oob_s2": int(s2.size),
        "n_flagged_mixed": n_flag1 + n_flag2,
        "n_separated": n_sep1 + n_sep2,
    }


def bootstrap_stratified(
    scores: np.ndarray,
    labels: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    stratum1_idx: np.ndarray,
    stratum2_idx: np.ndarray,
    n_boot: int,
    seed: int,
    grid: list[float] | None = None,
    target_spec: float = TARGET_SPECIFICITY,
) -> dict[str, Any]:
    """Per-stratum bootstrap calibration + combined stratified-router OOB cpWER.

    Draws ``n_boot`` bootstrap resamples from a single RNG stream (``seed``):
    stratum 1 first (size ``len(stratum1_idx)``), then stratum 2 (size
    ``len(stratum2_idx)``). On each (aligned) resample:

      - calibrate threshold_1 on stratum-1 in-bag, threshold_2 on stratum-2 in-bag
      - per-stratum OOB cpWER: route each stratum's OOB windows by its own threshold
      - combined OOB cpWER: pool both strata's OOB windows, each routed by its
        own stratum's threshold (the stratified router)

    Returns per-stratum threshold arrays, per-stratum OOB cpWER arrays, OOB sizes,
    and the combined OOB cpWER array. Deterministic for a given seed. Pure
    (no I/O)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mixed_cpwer = np.asarray(mixed_cpwer, dtype=float)
    sep_cpwer = np.asarray(sep_cpwer, dtype=float)
    s1 = np.asarray(stratum1_idx, dtype=int)
    s2 = np.asarray(stratum2_idx, dtype=int)
    n1, n2 = int(len(s1)), int(len(s2))

    sc1 = scores[s1]; lab1 = labels[s1]; mx1 = mixed_cpwer[s1]; sp1 = sep_cpwer[s1]
    sc2 = scores[s2]; lab2 = labels[s2]; mx2 = mixed_cpwer[s2]; sp2 = sep_cpwer[s2]

    rng = np.random.default_rng(seed)
    # Stratum 1 drawn first, then stratum 2, from the same stream -> deterministic.
    boot1 = rng.integers(0, n1, size=(n_boot, n1)) if n1 > 0 else np.empty((n_boot, 0), dtype=int)
    boot2 = rng.integers(0, n2, size=(n_boot, n2)) if n2 > 0 else np.empty((n_boot, 0), dtype=int)

    thr1 = np.empty(n_boot, dtype=float)
    thr2 = np.empty(n_boot, dtype=float)
    oob_cpwer_1 = np.empty(n_boot, dtype=float)
    oob_cpwer_2 = np.empty(n_boot, dtype=float)
    oob_cpwer_combined = np.empty(n_boot, dtype=float)
    n_oob_1 = np.empty(n_boot, dtype=int)
    n_oob_2 = np.empty(n_boot, dtype=int)

    all1 = np.arange(n1)
    all2 = np.arange(n2)

    for b in range(n_boot):
        # --- stratum 1
        if n1 > 0:
            ib1 = boot1[b]
            c1 = calibrate_threshold_at_spec(sc1[ib1], lab1[ib1], grid, target_spec)
            t1 = c1["threshold"]
            oob1_mask = ~np.isin(all1, np.unique(ib1))
            os1 = sc1[oob1_mask]; om1 = mx1[oob1_mask]; osp1 = sp1[oob1_mask]
            n_oob_1[b] = int(oob1_mask.sum())
            if os1.size == 0:
                oob_cpwer_1[b] = float("nan")
            else:
                f1 = os1 >= t1 - EPS
                oob_cpwer_1[b] = float(np.where(f1, om1, osp1).mean())
        else:
            t1 = float(grid[-1]) if grid else 1.0
            os1 = np.array([], dtype=float); om1 = np.array([], dtype=float); osp1 = np.array([], dtype=float)
            n_oob_1[b] = 0
            oob_cpwer_1[b] = float("nan")
        thr1[b] = t1

        # --- stratum 2
        if n2 > 0:
            ib2 = boot2[b]
            c2 = calibrate_threshold_at_spec(sc2[ib2], lab2[ib2], grid, target_spec)
            t2 = c2["threshold"]
            oob2_mask = ~np.isin(all2, np.unique(ib2))
            os2 = sc2[oob2_mask]; om2 = mx2[oob2_mask]; osp2 = sp2[oob2_mask]
            n_oob_2[b] = int(oob2_mask.sum())
            if os2.size == 0:
                oob_cpwer_2[b] = float("nan")
            else:
                f2 = os2 >= t2 - EPS
                oob_cpwer_2[b] = float(np.where(f2, om2, osp2).mean())
        else:
            t2 = float(grid[-1]) if grid else 1.0
            os2 = np.array([], dtype=float); om2 = np.array([], dtype=float); osp2 = np.array([], dtype=float)
            n_oob_2[b] = 0
            oob_cpwer_2[b] = float("nan")
        thr2[b] = t2

        # --- combined stratified router OOB
        comb = combined_oob_cpwer(os1, om1, osp1, t1, os2, om2, osp2, t2)
        oob_cpwer_combined[b] = comb["cpwer"]

    return {
        "n_boot": n_boot,
        "seed": seed,
        "n1": n1,
        "n2": n2,
        "thresholds_1": thr1,
        "thresholds_2": thr2,
        "oob_cpwer_1": oob_cpwer_1,
        "oob_cpwer_2": oob_cpwer_2,
        "oob_cpwer_combined": oob_cpwer_combined,
        "n_oob_1": n_oob_1,
        "n_oob_2": n_oob_2,
    }


def _summarise_threshold_distribution(thresholds: np.ndarray) -> dict[str, Any]:
    """Aggregate a bootstrap threshold distribution: median, mean, std, min/max,
    2.5/97.5 percentile interval + width, and modality (modes with >= 5%
    frequency via ``count_modes``)."""
    arr = np.asarray(thresholds, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "percentile_2_5": float("nan"), "percentile_97_5": float("nan"),
                "interval_width": float("nan"), "n_unique": 0, "n_modes_5pct": 0,
                "modes": []}
    lo, hi = percentile_interval(arr, 2.5, 97.5)
    md = count_modes(arr, MIN_MODE_FRACTION)
    return {
        "n": n,
        "median": round(float(np.median(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "percentile_2_5": round(lo, 6),
        "percentile_97_5": round(hi, 6),
        "interval_width": round(hi - lo, 6),
        "n_unique": md["n_unique"],
        "n_modes_5pct": md["n_modes"],
        "modes": md["modes"],
    }


def _summarise_oob_cpwer(values: np.ndarray, ref: float | None = None) -> dict[str, Any]:
    """Aggregate an OOB cpWER distribution: count of valid (non-nan) resamples,
    median, mean, min/max, 2.5/97.5 percentile, fraction below 1.10, and (if
    ``ref`` given) fraction below ``ref``."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    n_valid = int(valid.size)
    if n_valid == 0:
        out = {"n_valid": 0, "median": float("nan"), "mean": float("nan"),
               "min": float("nan"), "max": float("nan"),
               "percentile_2_5": float("nan"), "percentile_97_5": float("nan"),
               "frac_below_1_10": float("nan")}
        if ref is not None:
            out["frac_below_ref"] = float("nan")
        return out
    lo, hi = percentile_interval(valid, 2.5, 97.5)
    out = {
        "n_valid": n_valid,
        "median": round(float(np.median(valid)), 6),
        "mean": round(float(np.mean(valid)), 6),
        "min": round(float(np.min(valid)), 6),
        "max": round(float(np.max(valid)), 6),
        "percentile_2_5": round(lo, 6),
        "percentile_97_5": round(hi, 6),
        "frac_below_1_10": round(float(np.mean(valid < 1.10)), 6),
    }
    if ref is not None:
        out["frac_below_ref"] = round(float(np.mean(valid < ref)), 6)
    return out


def _bootstrap_pooled(
    scores: np.ndarray,
    labels: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    n_boot: int,
    seed: int,
    grid: list[float] | None = None,
    target_spec: float = TARGET_SPECIFICITY,
) -> dict[str, Any]:
    """Pooled bootstrap (RQ44 recipe) at matched B and seed, for an internal
    apples-to-apples comparison to the stratified router. Not the primary
    baseline (RQ44's published B=10000 numbers are cited separately) but lets
    the within-script comparison hold B and seed fixed."""
    n = int(len(scores))
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, n, size=(n_boot, n))
    thr = np.empty(n_boot, dtype=float)
    oob = np.empty(n_boot, dtype=float)
    n_oob = np.empty(n_boot, dtype=int)
    all_idx = np.arange(n)
    for b in range(n_boot):
        idx = boot[b]
        cal = calibrate_threshold_at_spec(scores[idx], labels[idx], grid, target_spec)
        t = cal["threshold"]
        thr[b] = t
        oob_mask = ~np.isin(all_idx, np.unique(idx))
        os_ = scores[oob_mask]; om = mixed_cpwer[oob_mask]; op = sep_cpwer[oob_mask]
        n_oob[b] = int(oob_mask.sum())
        if os_.size == 0:
            oob[b] = float("nan")
        else:
            f = os_ >= t - EPS
            oob[b] = float(np.where(f, om, op).mean())
    return {"thresholds": thr, "oob_cpwer": oob, "n_oob": n_oob}


# --------------------------------------------------------------------- driver
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    windows = data["windows"]
    n = len(windows)

    # Per-window signals.
    lang_ent = np.array([max_across_speakers(w) for w in windows], dtype=float)
    mixed_cpwer = np.array([float(w["always_mixed_cpwer"]) for w in windows], dtype=float)
    sep_cpwer = np.array([float(w["always_separated_cpwer"]) for w in windows], dtype=float)
    active_spk = np.array([count_active_speakers(w.get("separated_text_per_speaker", {})) for w in windows], dtype=int)
    labels = (sep_cpwer > CATASTROPHIC_CPWER).astype(int)  # 1 = hallucinated
    n_hall = int(labels.sum())
    n_clean = int((labels == 0).sum())

    # ------------------------------------------------------- stratify
    s1_idx, s2_idx = stratify_by_speaker_count(active_spk, SPEAKER_SPLIT)
    n1, n2 = int(len(s1_idx)), int(len(s2_idx))
    hall1 = int(labels[s1_idx].sum())
    hall2 = int(labels[s2_idx].sum())

    # Mode S windows (RQ19/RQ38): w22 and w30 are low-entropy hallucinated.
    mode_s_ids = {22, 30}
    mode_s_in_s1 = [int(w["window_id"]) for w in windows
                    if w["window_id"] in mode_s_ids
                    and int(w["window_id"]) in set(int(w["window_id"]) for w in [windows[i] for i in s1_idx])]
    # (stratum1_idx are positional indices into `windows`; map to window_ids)
    s1_window_ids = {int(windows[i]["window_id"]) for i in s1_idx}
    s2_window_ids = {int(windows[i]["window_id"]) for i in s2_idx}
    mode_s_in_s1 = sorted(mode_s_ids & s1_window_ids)

    # --------------------------------------------- in-sample pooled reproduction (RQ44 ref)
    in_sample = calibrate_threshold_at_spec(lang_ent, labels)
    in_flag = lang_ent >= in_sample["threshold"] - EPS
    in_selected = np.where(in_flag, mixed_cpwer, sep_cpwer)
    in_sample_cpwer = float(in_selected.mean())

    # ------------------------------------------------------- per-stratum in-sample
    in_s1 = calibrate_threshold_at_spec(lang_ent[s1_idx], labels[s1_idx])
    in_s2 = calibrate_threshold_at_spec(lang_ent[s2_idx], labels[s2_idx])

    # ------------------------------------------------------- pooled bootstrap (matched B/seed)
    pooled = _bootstrap_pooled(lang_ent, labels, mixed_cpwer, sep_cpwer, N_BOOT, SEED)
    pooled_thr_dist = _summarise_threshold_distribution(pooled["thresholds"])
    pooled_oob_dist = _summarise_oob_cpwer(pooled["oob_cpwer"], ref=RQ44_POOLED_MEDIAN_OOB_CPWER)

    # ------------------------------------------------------- stratified bootstrap
    boot = bootstrap_stratified(
        lang_ent, labels, mixed_cpwer, sep_cpwer, s1_idx, s2_idx,
        N_BOOT, SEED,
    )
    thr1_dist = _summarise_threshold_distribution(boot["thresholds_1"])
    thr2_dist = _summarise_threshold_distribution(boot["thresholds_2"])
    oob1_dist = _summarise_oob_cpwer(boot["oob_cpwer_1"])
    oob2_dist = _summarise_oob_cpwer(boot["oob_cpwer_2"])
    combined_oob_dist = _summarise_oob_cpwer(boot["oob_cpwer_combined"], ref=RQ44_POOLED_MEDIAN_OOB_CPWER)

    # ------------------------------------------------------------ hypotheses
    h49a_supported = thr1_dist["n_modes_5pct"] <= H49A_MAX_MODES
    h49b_supported = thr2_dist["n_modes_5pct"] <= H49B_MAX_MODES
    h49c_supported = (not math.isnan(combined_oob_dist["median"])) and combined_oob_dist["median"] < H49C_MAX_CPWER

    summary: dict[str, Any] = {
        "label": "experimental/frontier",
        "rq": "RQ49: Speaker-count stratified threshold for reduced modality",
        "builds_on": {
            "RQ13": "results/frontier/diverse_hallucination_detector/ (PR #904)",
            "RQ16": "results/frontier/corrected_router_simulation/ (PR #912)",
            "RQ25": "results/frontier/out_of_sample_router/ (PR #929)",
            "RQ38": "results/frontier/speaker_count_effect/ (PR #948)",
            "RQ44": "results/frontier/bootstrap_threshold_stability/ (PR #963)",
        },
        "source_data": str(SRC_JSON.relative_to(PROJECT_ROOT)),
        "source_label": "external/sanity-check",
        "method": (
            "reanalysis only (no Whisper / no ASR run). Active speakers = count of "
            "non-empty separated_text_per_speaker values (RQ38 definition). Stratify "
            "the 77 windows into <=2 active speakers (stratum 1, where Mode S occurs) "
            "and >2 active speakers (stratum 2). Per stratum: B=2000 bootstrap "
            "resamples (seed=42, n=stratum_size with replacement); calibrate lang-id "
            "entropy threshold maximising sensitivity at >=90% specificity (RQ44 rule) "
            "on in-bag; evaluate corrected-router cpWER on out-of-bag. Stratified "
            "router: aligned resamples, calibrate a threshold per stratum, evaluate "
            "COMBINED OOB cpWER (each OOB window routed by its own stratum's threshold)."
        ),
        "meeting_id": data["meeting_id"],
        "n_windows": n,
        "n_hallucinated": n_hall,
        "n_clean": n_clean,
        "hallucination_label_rule": "always_separated_cpwer > 1.0",
        "active_speakers_rule": "count of non-empty separated_text_per_speaker values (RQ38)",
        "routing_rule": (
            "lang_id_entropy >= threshold -> route MIXED (always_mixed_cpwer); "
            "else route SEPARATED (always_separated_cpwer). HIGH lang-id entropy = "
            "diverse multilingual gibberish = hallucination (RQ13/RQ16/RQ25/RQ44)."
        ),
        "calibration_rule": (
            "sweep threshold 0.0-2.0 bits in 0.01 steps; select threshold maximising "
            "sensitivity at >= 90% specificity. Tie-breaker: higher specificity, then "
            "lower threshold."
        ),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "resample_size_stratum1": n1,
            "resample_size_stratum2": n2,
            "expected_oob_size_stratum1": round(n1 * ((1 - 1 / n1) ** n1), 4) if n1 > 0 else 0.0,
            "expected_oob_size_stratum2": round(n2 * ((1 - 1 / n2) ** n2), 4) if n2 > 0 else 0.0,
        },
        "stratification": {
            "split": SPEAKER_SPLIT,
            "stratum1": {
                "rule": "active_speakers <= 2",
                "n_windows": n1,
                "n_hallucinated": hall1,
                "hallucination_rate": round(hall1 / n1, 6) if n1 > 0 else float("nan"),
                "window_ids": sorted(s1_window_ids),
            },
            "stratum2": {
                "rule": "active_speakers > 2",
                "n_windows": n2,
                "n_hallucinated": hall2,
                "hallucination_rate": round(hall2 / n2, 6) if n2 > 0 else float("nan"),
                "window_ids": sorted(s2_window_ids),
            },
            "mode_s_windows": sorted(mode_s_ids),
            "mode_s_in_stratum1": mode_s_in_s1,
        },
        "in_sample_reproduction": {
            "RQ44_threshold_reference": RQ44_IN_SAMPLE_THRESHOLD,
            "pooled_threshold": in_sample["threshold"],
            "pooled_sensitivity": in_sample["sensitivity"],
            "pooled_specificity": in_sample["specificity"],
            "pooled_cpwer": round(in_sample_cpwer, 6),
            "stratum1_threshold": in_s1["threshold"],
            "stratum1_sensitivity": in_s1["sensitivity"],
            "stratum1_specificity": in_s1["specificity"],
            "stratum2_threshold": in_s2["threshold"],
            "stratum2_sensitivity": in_s2["sensitivity"],
            "stratum2_specificity": in_s2["specificity"],
            "note": "Pooled reproduces RQ44/RQ25's 0.38 in-sample threshold on all 77 windows.",
        },
        "rq44_pooled_reference": {
            "n_boot": 10000,
            "n_unique": RQ44_POOLED_N_UNIQUE,
            "n_modes_5pct": RQ44_POOLED_N_MODES_5PCT,
            "interval_width": RQ44_POOLED_WIDTH,
            "median_oob_cpwer": RQ44_POOLED_MEDIAN_OOB_CPWER,
            "source": "results/frontier/bootstrap_threshold_stability/bootstrap_threshold_results.json",
        },
        "pooled_bootstrap_matched": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "threshold_distribution": pooled_thr_dist,
            "oob_cpwer_distribution": pooled_oob_dist,
            "note": "Pooled RQ44 recipe at matched B=2000, seed=42 for within-script comparison.",
        },
        "stratum1_threshold_distribution": thr1_dist,
        "stratum2_threshold_distribution": thr2_dist,
        "stratum1_oob_cpwer_distribution": oob1_dist,
        "stratum2_oob_cpwer_distribution": oob2_dist,
        "combined_stratified_oob_cpwer_distribution": combined_oob_dist,
        "hypothesis_verdicts": {
            "H49a": {
                "statement": "<=2-speaker threshold distribution has <= 3 modes (>= 5% frequency)",
                "n_modes_5pct": thr1_dist["n_modes_5pct"],
                "kill_threshold": H49A_MAX_MODES,
                "rq44_pooled_n_modes_5pct": RQ44_POOLED_N_MODES_5PCT,
                "supported": bool(h49a_supported),
            },
            "H49b": {
                "statement": ">2-speaker threshold distribution has <= 2 modes (>= 5% frequency)",
                "n_modes_5pct": thr2_dist["n_modes_5pct"],
                "kill_threshold": H49B_MAX_MODES,
                "supported": bool(h49b_supported),
            },
            "H49c": {
                "statement": "stratified router median OOB cpWER < RQ44's 1.056",
                "median_oob_cpwer": combined_oob_dist["median"],
                "kill_threshold": H49C_MAX_CPWER,
                "rq44_pooled_median_oob_cpwer_exact": RQ44_POOLED_MEDIAN_OOB_CPWER,
                "rq44_pooled_median_oob_cpwer_rounded": RQ44_POOLED_MEDIAN_OOB_CPWER_ROUNDED,
                "pooled_matched_b2000_median_oob_cpwer": pooled_oob_dist["median"],
                "frac_combined_below_rq44": combined_oob_dist.get("frac_below_ref", float("nan")),
                "supported": bool(h49c_supported),
                "substantive_note": (
                    "H49c is supported only by a sub-0.001 rounding margin: the combined "
                    "median (1.055556) is numerically IDENTICAL to RQ44's exact pooled "
                    "median (1.055556) and slightly higher than the matched-B pooled "
                    "median (%.6f). Stratification provides no substantive OOB cpWER "
                    "improvement."
                ) % pooled_oob_dist["median"],
            },
        },
    }

    # ----------------------------------------------------------- per-bootstrap arrays
    summary["per_bootstrap"] = {
        "thresholds_1": [round(float(t), 6) for t in boot["thresholds_1"]],
        "thresholds_2": [round(float(t), 6) for t in boot["thresholds_2"]],
        "oob_cpwer_1": [round(float(c), 6) if not math.isnan(float(c)) else None for c in boot["oob_cpwer_1"]],
        "oob_cpwer_2": [round(float(c), 6) if not math.isnan(float(c)) else None for c in boot["oob_cpwer_2"]],
        "oob_cpwer_combined": [round(float(c), 6) if not math.isnan(float(c)) else None for c in boot["oob_cpwer_combined"]],
        "n_oob_1": [int(x) for x in boot["n_oob_1"]],
        "n_oob_2": [int(x) for x in boot["n_oob_2"]],
    }

    # ----------------------------------------------------------- per-stratum CSV
    csv_fields = [
        "stratum", "n_windows", "n_hallucinated", "hallucination_rate",
        "thr_median", "thr_p2_5", "thr_p97_5", "thr_width",
        "thr_n_unique", "thr_n_modes_5pct",
        "oob_cpwer_median", "oob_cpwer_mean", "oob_cpwer_frac_below_1_10",
    ]

    def _csv_row(name: str, n_w: int, n_h: int, thr_dist: dict, oob_dist: dict) -> dict:
        return {
            "stratum": name,
            "n_windows": n_w,
            "n_hallucinated": n_h,
            "hallucination_rate": round(n_h / n_w, 6) if n_w > 0 else "",
            "thr_median": thr_dist.get("median", ""),
            "thr_p2_5": thr_dist.get("percentile_2_5", ""),
            "thr_p97_5": thr_dist.get("percentile_97_5", ""),
            "thr_width": thr_dist.get("interval_width", ""),
            "thr_n_unique": thr_dist.get("n_unique", ""),
            "thr_n_modes_5pct": thr_dist.get("n_modes_5pct", ""),
            "oob_cpwer_median": oob_dist.get("median", ""),
            "oob_cpwer_mean": oob_dist.get("mean", ""),
            "oob_cpwer_frac_below_1_10": oob_dist.get("frac_below_1_10", ""),
        }

    rows = [
        _csv_row("pooled_matched_b2000", n, n_hall, pooled_thr_dist, pooled_oob_dist),
        _csv_row("stratum1_le2", n1, hall1, thr1_dist, oob1_dist),
        _csv_row("stratum2_gt2", n2, hall2, thr2_dist, oob2_dist),
        _csv_row("combined_stratified", n, n_hall, thr1_dist, combined_oob_dist),
    ]
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
    print(f"=== RQ49: Speaker-count stratified threshold (AISHELL-4, {n} windows) ===")
    print(f"Label: experimental/frontier  |  Source: {SRC_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Hallucination label: always_separated_cpwer > 1.0 -> {n_hall} hall / {n_clean} clean")
    print(f"Active speakers = count of non-empty separated transcripts (RQ38).")
    print()
    print("Stratification (split at <=2 vs >2 active speakers):")
    print(f"  Stratum 1 (<=2): n={n1}  halluc={hall1}  rate={hall1/n1:.4f}" if n1 else "  Stratum 1: empty")
    print(f"  Stratum 2 (>2):  n={n2}  halluc={hall2}  rate={hall2/n2:.4f}" if n2 else "  Stratum 2: empty")
    print(f"  Mode S windows (w22, w30) in stratum 1: {mode_s_in_s1}")
    print()
    print("In-sample reproduction (calibrate + evaluate on all 77, RQ44 reference):")
    print(f"  pooled threshold         : {in_sample['threshold']:.4f}  (RQ44/RQ25 reported 0.38)")
    print(f"  pooled corrected cpWER   : {in_sample_cpwer:.6f}")
    print(f"  stratum1 threshold       : {in_s1['threshold']:.4f}  (sens={in_s1['sensitivity']:.4f}, spec={in_s1['specificity']:.4f})")
    print(f"  stratum2 threshold       : {in_s2['threshold']:.4f}  (sens={in_s2['sensitivity']:.4f}, spec={in_s2['specificity']:.4f})")
    print()
    print(f"Bootstrap: B={N_BOOT}, seed={SEED}")
    print()
    print("Pooled bootstrap (matched B=2000, seed=42; RQ44 published B=10000 ref in parens):")
    print(f"  threshold median / width : {pooled_thr_dist['median']:.4f} / {pooled_thr_dist['interval_width']:.4f}  "
          f"(RQ44: width={RQ44_POOLED_WIDTH})")
    print(f"  threshold n_unique       : {pooled_thr_dist['n_unique']}  (RQ44: {RQ44_POOLED_N_UNIQUE})")
    print(f"  threshold modes (>=5%)   : {pooled_thr_dist['n_modes_5pct']}  (RQ44: {RQ44_POOLED_N_MODES_5PCT})")
    print(f"  OOB cpWER median         : {pooled_oob_dist['median']:.6f}  (RQ44: {RQ44_POOLED_MEDIAN_OOB_CPWER})")
    print()
    print("Stratum 1 (<=2 active speakers) threshold distribution:")
    _print_thr_dist(thr1_dist)
    print("  OOB cpWER: median=%.6f  mean=%.6f  frac<1.10=%.4f" % (
        oob1_dist["median"], oob1_dist["mean"], oob1_dist["frac_below_1_10"]))
    print()
    print("Stratum 2 (>2 active speakers) threshold distribution:")
    _print_thr_dist(thr2_dist)
    print("  OOB cpWER: median=%.6f  mean=%.6f  frac<1.10=%.4f" % (
        oob2_dist["median"], oob2_dist["mean"], oob2_dist["frac_below_1_10"]))
    print()
    print("Combined stratified-router OOB cpWER:")
    print(f"  n valid resamples        : {combined_oob_dist['n_valid']} / {N_BOOT}")
    print(f"  median                   : {combined_oob_dist['median']:.6f}  (RQ44 pooled: {RQ44_POOLED_MEDIAN_OOB_CPWER})")
    print(f"  mean                     : {combined_oob_dist['mean']:.6f}")
    print(f"  2.5 / 97.5 pct           : [{combined_oob_dist['percentile_2_5']:.4f}, {combined_oob_dist['percentile_97_5']:.4f}]")
    print(f"  frac < 1.10              : {combined_oob_dist['frac_below_1_10']:.4f}")
    print(f"  frac < RQ44 (1.056)      : {combined_oob_dist.get('frac_below_ref', float('nan')):.4f}")
    print()
    print("Hypothesis verdicts:")
    print(f"  H49a (stratum1 modes <= 3): {'SUPPORTED' if h49a_supported else 'KILLED'}  "
          f"(modes_5pct={thr1_dist['n_modes_5pct']}, kill>{H49A_MAX_MODES}; RQ44 pooled={RQ44_POOLED_N_MODES_5PCT})")
    print(f"  H49b (stratum2 modes <= 2): {'SUPPORTED' if h49b_supported else 'KILLED'}  "
          f"(modes_5pct={thr2_dist['n_modes_5pct']}, kill>{H49B_MAX_MODES})")
    print(f"  H49c (combined OOB cpWER < 1.056): {'SUPPORTED' if h49c_supported else 'KILLED'}  "
          f"(median={combined_oob_dist['median']:.6f})")
    print()
    print(f"Wrote: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}")


def _print_thr_dist(d: dict[str, Any]) -> None:
    print(f"  threshold median / width : {d['median']:.4f} / {d['interval_width']:.4f}")
    print(f"  2.5 / 97.5 pct           : [{d['percentile_2_5']:.4f}, {d['percentile_97_5']:.4f}]")
    print(f"  n_unique                 : {d['n_unique']}")
    print(f"  modes (>=5% frequency)   : {d['n_modes_5pct']}")
    for m in d["modes"]:
        print(f"    threshold={m['threshold']:.4f}  count={m['count']}  frac={m['fraction']:.4f}")


if __name__ == "__main__":
    main()
