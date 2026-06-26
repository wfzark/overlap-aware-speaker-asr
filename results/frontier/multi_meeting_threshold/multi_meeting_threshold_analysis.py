"""RQ45: Multi-meeting threshold stability simulation.

REANALYSIS ONLY -- no Whisper / no ASR model is run. This script reads the
existing AISHELL-4 external-validation results
(``results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json``,
label ``external/sanity-check``, PR #890) and simulates a larger calibration
corpus by bootstrap-resampling the 77 AISHELL-4 windows at different sample
sizes.

Motivation (RQ44, PR #963)
--------------------------
RQ44 showed the lang-id entropy threshold distribution is 6-modal over
[0.01, 0.95] when bootstrap-resampling the 77 AISHELL-4 windows (H44b KILLED,
interval width 0.94). The 6-modality is driven by the small sample size (n=77)
-- specifically the 2 Mode S windows and the high-entropy clean windows with
tied cpWER. RQ45 asks: does the threshold distribution converge to unimodal as
the calibration sample size increases? We simulate a larger calibration corpus
by bootstrap-resampling the 77 AISHELL-4 windows at different sample sizes
(n=77, 154, 308, 616, 1232) and measuring the threshold distribution's
modality, percentile interval width, and OOB cpWER at each size.

Method (RQ45)
-------------
For each sample size n in {77, 154, 308, 616, 1232}:
  1. Draw B=2000 bootstrap resamples of size n (with replacement from the 77
     windows; duplicates allowed).
  2. On each resample, calibrate the lang-id entropy threshold (>=90%
     specificity, max sensitivity) using the same rule as RQ44 (sweep grid
     {0.00, 0.01, ..., 2.00}, tie-breaker: higher specificity then lower
     threshold).
  3. Record the calibrated threshold.
  4. Compute OOB cpWER on the out-of-bag windows (windows NOT drawn in the
     resample).

For each n, report: threshold distribution (median, mean, std, 2.5/97.5
percentile interval, width, number of modes with >=5% frequency, mode table),
OOB cpWER distribution (median, mean, 2.5/97.5 percentile, fraction < 1.10).

Detector (RQ44 verbatim)
------------------------
The lang-id entropy detector from RQ44 is used verbatim. Shannon entropy (bits)
over Unicode script-category distribution of the text, MAX-aggregated across
speakers. Hallucination label: ``always_separated_cpwer > 1.0`` (37
hallucinated, 40 clean).

Pre-registered hypotheses (issue for RQ45)
------------------------------------------
- H45a: At n=616 (8x the original), the threshold distribution is unimodal
        (<= 2 distinct thresholds with >=5% frequency each). Kill: > 2 modes
        with >=5% frequency.
- H45b: At n=616, the 2.5/97.5 percentile interval width < 0.20 (RQ44's H44b
        kill threshold). Kill: >= 0.20.
- H45c: At n=616, the median OOB cpWER < 1.05 (tighter than RQ44's 1.056).
        Kill: >= 1.05.

This script is pure reanalysis (numpy + stdlib only; scipy / sklearn / Whisper
are NOT required). The detector primitives (``script_category``,
``language_id_entropy``, ``max_across_speakers``) and the calibration rule
(``calibrate_threshold_at_spec``) are lifted verbatim from RQ44 so thresholds
are directly comparable.

Label: experimental/frontier. Builds on RQ13 (PR #904), RQ16 (PR #912), RQ25
(PR #929), and RQ44 (PR #963).
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
OUT_DIR = PROJECT_ROOT / "results" / "frontier" / "multi_meeting_threshold"
OUT_CSV = OUT_DIR / "multi_meeting_threshold_results.csv"
OUT_JSON = OUT_DIR / "multi_meeting_threshold_results.json"

# ------------------------------------------------------------------ constants
CATASTROPHIC_CPWER = 1.0    # cpWER > 1.0 => insertions dominate (hallucination label)
TARGET_SPECIFICITY = 0.90   # calibrate the threshold to >= 90% specificity
THRESHOLD_GRID = [round(0.01 * i, 2) for i in range(0, 201)]  # 0.00, 0.01, ..., 2.00
# Sample sizes: 1x, 2x, 4x, 8x, 16x the original n=77. Multiples are exact to
# match the pre-registered hypothesis anchors (n=616 = 8x). The 2x/4x/16x are
# rounded to integers (77*2=154, 77*4=308, 77*8=616, 77*16=1232).
SAMPLE_SIZES = [77, 154, 308, 616, 1232]
N_BOOT = 2000
SEED = 42
EPS = 1e-9
MODE_MIN_FRACTION = 0.05   # a "mode" is a distinct threshold with >= 5% frequency

# RQ25's in-sample threshold on the 77 windows (0.01 grid) = 0.38; RQ44's
# bootstrap median on n=77 (B=10000) was also 0.38. The smoke test reproduces
# this on n=77 with B=2000 (the bootstrap median should still be 0.38 since the
# dominant mode is 0.38).
RQ25_IN_SAMPLE_THRESHOLD = 0.38
RQ44_BOOTSTRAP_MEDIAN_THRESHOLD = 0.38
RQ44_INTERVAL_WIDTH = 0.94
RQ44_OOB_CPWER_MEDIAN = 1.056

# Hypothesis anchors (all evaluated at n=616).
H45_ANCHOR_N = 616
H45A_MAX_MODES = 2          # unimodal: <= 2 distinct thresholds with >=5% frequency
H45B_MAX_WIDTH = 0.20       # same kill threshold as RQ44's H44b
H45C_MAX_CPWER = 1.05       # tighter than RQ44's 1.056 OOB median


# ------------------------------------------------------------- script detection
def script_category(ch: str) -> str:
    """Map a character to a coarse Unicode script category (RQ13/RQ16/RQ25/RQ44 verbatim).

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


# --------------------------------------------------------------- pure helpers
def bootstrap_resample(
    n_population: int, n_sample: int, n_boot: int, seed: int
) -> np.ndarray:
    """Return an ``(n_boot, n_sample)`` int array of bootstrap resample indices.

    Each row is an independent bootstrap resample: ``n_sample`` indices drawn
    with replacement from ``{0, ..., n_population-1}``. ``n_sample`` MAY differ
    from ``n_population`` (RQ45 simulates larger calibration corpora by drawing
    n_sample > n_population). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_population, size=(n_boot, n_sample))


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
    confusion counts. Faithful verbatim copy of RQ44's
    ``calibrate_threshold_at_spec`` so thresholds are directly comparable.
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


def count_modes(
    thresholds: np.ndarray, min_fraction: float = MODE_MIN_FRACTION
) -> dict[str, Any]:
    """Count the distinct thresholds whose frequency is >= ``min_fraction`` of the
    total number of resamples. A "mode" is a distinct threshold value with at
    least ``min_fraction`` (default 5%) of the bootstrap mass.

    Returns the number of modes (``n_modes``), the list of modes (threshold,
    count, fraction) sorted by descending count then ascending threshold, the
    number of unique values, and the dominant mode.

    H45a uses ``n_modes <= 2`` as the unimodality criterion (the pre-registered
    kill is ``n_modes > 2``).
    """
    arr = np.asarray(thresholds, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n_modes": 0, "modes": [], "n_unique": 0, "dominant": float("nan"),
                "dominant_fraction": 0.0}
    uniq, counts = np.unique(arr, return_counts=True)
    fractions = counts / n
    mode_mask = fractions >= min_fraction - EPS
    mode_idx = np.where(mode_mask)[0]
    # Sort modes by descending count, then ascending threshold (deterministic).
    mode_order = mode_idx[np.argsort(-counts[mode_idx], kind="stable")]
    modes = [
        {"threshold": float(uniq[i]), "count": int(counts[i]),
         "fraction": float(fractions[i])}
        for i in mode_order
    ]
    dominant_idx = int(np.argmax(counts))
    return {
        "n_modes": int(len(modes)),
        "modes": modes,
        "n_unique": int(uniq.size),
        "dominant": float(uniq[dominant_idx]),
        "dominant_fraction": float(fractions[dominant_idx]),
    }


def percentile_interval_width(
    values: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> float:
    """Return the width (``hi_pct - lo_pct``) of the percentile interval of
    ``values`` using numpy's default linear interpolation. Returns ``nan`` for
    empty input or for inputs containing only NaN."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, hi) - np.percentile(arr, lo))


def percentile_interval(
    values: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    """Return the ``(lo, hi)`` percentile interval of ``values`` (NaN-filtered).
    Returns ``(nan, nan)`` for empty input. Used for reporting the interval
    endpoints alongside ``percentile_interval_width``."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def out_of_bag_cpwer(
    scores: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    threshold: float,
    in_bag_idx: np.ndarray,
) -> dict[str, Any]:
    """Compute the corrected-router cpWER on the out-of-bag (OOB) windows --
    those NOT present in the bootstrap resample ``in_bag_idx`` -- at the given
    ``threshold``.

    Routing: ``score >= threshold`` => route MIXED (``mixed_cpwer``); else
    SEPARATED (``sep_cpwer``). Returns a dict with the mean selected cpWER over
    the OOB windows (``nan`` if there are none), the OOB size, and the
    mixed/separated decision counts. Verbatim copy of RQ44's helper.

    Note: for ``n_sample > n_population`` (RQ45's larger simulated corpora), the
    OOB set shrinks rapidly -- at n_sample=616 with n_population=77, the
    expected OOB size is ~0.025 windows, so most resamples have an empty OOB
    set and return ``cpwer=nan``. This is an inherent limitation of simulating
    a larger corpus by resampling a fixed 77-window pool; see FINDINGS.md."""
    n = len(scores)
    all_idx = np.arange(n)
    in_bag_set = np.unique(np.asarray(in_bag_idx, dtype=int))
    oob_mask = ~np.isin(all_idx, in_bag_set)
    oob_scores = scores[oob_mask]
    oob_mixed = mixed_cpwer[oob_mask]
    oob_sep = sep_cpwer[oob_mask]
    n_oob = int(oob_mask.sum())
    if n_oob == 0:
        return {"cpwer": float("nan"), "n_oob": 0,
                "n_flagged_mixed": 0, "n_separated": 0}
    flagged = oob_scores >= threshold - EPS
    selected = np.where(flagged, oob_mixed, oob_sep)
    return {
        "cpwer": float(selected.mean()),
        "n_oob": n_oob,
        "n_flagged_mixed": int(flagged.sum()),
        "n_separated": int((~flagged).sum()),
    }


def expected_oob_size(n_population: int, n_sample: int) -> float:
    """Expected number of OOB windows when drawing ``n_sample`` indices with
    replacement from ``n_population`` windows:
    ``n_population * (1 - 1/n_population) ** n_sample``.

    For n_population=77: n_sample=77 -> ~28.14, n_sample=616 -> ~0.025,
    n_sample=1232 -> ~7.7e-6. The rapid shrinkage is why OOB cpWER becomes
    undefined for most resamples at n_sample >= 308."""
    if n_population <= 0:
        return 0.0
    return float(n_population * ((1 - 1 / n_population) ** n_sample))


# --------------------------------------------------------------------- driver
def run_one_sample_size(
    n_sample: int,
    lang_ent: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Run the full bootstrap procedure for a single sample size ``n_sample``.

    Returns a dict with the per-resample arrays (thresholds, oob_cpwer, n_oob)
    and the aggregated distributions. The population is always the 77
    AISHELL-4 windows; ``n_sample`` is the resample size (may exceed 77)."""
    n_pop = len(lang_ent)
    boot_idx = bootstrap_resample(n_pop, n_sample, n_boot, seed)  # (n_boot, n_sample)
    boot_thresholds = np.empty(n_boot, dtype=float)
    boot_oob_cpwer = np.empty(n_boot, dtype=float)
    boot_n_oob = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        idx = boot_idx[b]
        cal = calibrate_threshold_at_spec(lang_ent[idx], labels[idx])
        thr = cal["threshold"]
        boot_thresholds[b] = thr
        oob = out_of_bag_cpwer(lang_ent, mixed_cpwer, sep_cpwer, thr, idx)
        boot_oob_cpwer[b] = oob["cpwer"]
        boot_n_oob[b] = oob["n_oob"]

    # Threshold distribution.
    thr_median = float(np.median(boot_thresholds))
    thr_mean = float(np.mean(boot_thresholds))
    thr_std = float(np.std(boot_thresholds))
    thr_lo, thr_hi = percentile_interval(boot_thresholds, 2.5, 97.5)
    thr_width = percentile_interval_width(boot_thresholds, 2.5, 97.5)
    thr_min = float(np.min(boot_thresholds))
    thr_max = float(np.max(boot_thresholds))
    thr_modes = count_modes(boot_thresholds, MODE_MIN_FRACTION)

    # OOB cpWER distribution.
    valid_mask = ~np.isnan(boot_oob_cpwer)
    valid_oob = boot_oob_cpwer[valid_mask]
    n_valid = int(valid_mask.sum())
    if n_valid > 0:
        oob_median = float(np.median(valid_oob))
        oob_mean = float(np.mean(valid_oob))
        oob_lo, oob_hi = percentile_interval(valid_oob, 2.5, 97.5)
        oob_min = float(np.min(valid_oob))
        oob_max = float(np.max(valid_oob))
        oob_p25, oob_p75 = percentile_interval(valid_oob, 25, 75)
        oob_frac_below_110 = float(np.mean(valid_oob < 1.10))
    else:
        oob_median = float("nan")
        oob_mean = float("nan")
        oob_lo, oob_hi = float("nan"), float("nan")
        oob_min, oob_max = float("nan"), float("nan")
        oob_p25, oob_p75 = float("nan"), float("nan")
        oob_frac_below_110 = float("nan")
    oob_n_oob_mean = float(np.mean(boot_n_oob))  # includes 0s

    return {
        "n_sample": n_sample,
        "n_population": n_pop,
        "n_boot": n_boot,
        "seed": seed,
        "expected_oob_size": expected_oob_size(n_pop, n_sample),
        "per_bootstrap": {
            "thresholds": boot_thresholds,
            "oob_cpwer": boot_oob_cpwer,
            "n_oob": boot_n_oob,
        },
        "threshold_distribution": {
            "median": thr_median,
            "mean": thr_mean,
            "std": thr_std,
            "min": thr_min,
            "max": thr_max,
            "percentile_2_5": thr_lo,
            "percentile_97_5": thr_hi,
            "interval_width": thr_width,
            "n_modes": thr_modes["n_modes"],
            "modes": thr_modes["modes"],
            "n_unique": thr_modes["n_unique"],
            "dominant": thr_modes["dominant"],
            "dominant_fraction": thr_modes["dominant_fraction"],
        },
        "oob_cpwer_distribution": {
            "n_valid_resamples": n_valid,
            "mean_oob_size": oob_n_oob_mean,
            "median": oob_median,
            "mean": oob_mean,
            "min": oob_min,
            "max": oob_max,
            "percentile_2_5": oob_lo,
            "percentile_97_5": oob_hi,
            "iqr": [oob_p25, oob_p75],
            "frac_below_1_10": oob_frac_below_110,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    windows = data["windows"]
    n_pop = len(windows)

    # Per-window signals.
    lang_ent = np.array([max_across_speakers(w) for w in windows], dtype=float)
    mixed_cpwer = np.array([float(w["always_mixed_cpwer"]) for w in windows], dtype=float)
    sep_cpwer = np.array([float(w["always_separated_cpwer"]) for w in windows], dtype=float)
    labels = (sep_cpwer > CATASTROPHIC_CPWER).astype(int)  # 1 = hallucinated
    n_hall = int(labels.sum())
    n_clean = int((labels == 0).sum())

    # --------------------------------------------- in-sample baseline (RQ25/RQ44 reproduction)
    in_sample = calibrate_threshold_at_spec(lang_ent, labels)
    in_flag = lang_ent >= in_sample["threshold"] - EPS
    in_selected = np.where(in_flag, mixed_cpwer, sep_cpwer)
    in_sample_cpwer = float(in_selected.mean())

    # ----------------------------------------------------------------- per-n bootstrap
    per_n: list[dict[str, Any]] = []
    for n_sample in SAMPLE_SIZES:
        # Use a per-n seed derived from the base SEED so each n is deterministic
        # but the resamples for different n are independent (avoids the n=77
        # resample being a prefix of the n=154 resample, which would couple
        # them artificially). The base seed (42) is used for n=77 to keep the
        # n=77 result directly comparable to RQ44's seed-42 resampling.
        n_seed = SEED if n_sample == SAMPLE_SIZES[0] else SEED + n_sample
        res = run_one_sample_size(
            n_sample, lang_ent, mixed_cpwer, sep_cpwer, labels, N_BOOT, n_seed
        )
        per_n.append(res)

    # ------------------------------------------------------------ hypotheses (anchored at n=616)
    anchor = next(r for r in per_n if r["n_sample"] == H45_ANCHOR_N)
    anchor_thr = anchor["threshold_distribution"]
    anchor_oob = anchor["oob_cpwer_distribution"]
    h45a_supported = anchor_thr["n_modes"] <= H45A_MAX_MODES
    h45b_supported = (
        not math.isnan(anchor_thr["interval_width"])
        and anchor_thr["interval_width"] < H45B_MAX_WIDTH
    )
    h45c_supported = (
        not math.isnan(anchor_oob["median"])
        and anchor_oob["median"] < H45C_MAX_CPWER
    )

    # ------------------------------------------------------------ summary
    summary: dict[str, Any] = {
        "label": "experimental/frontier",
        "rq": "RQ45: Multi-meeting threshold stability simulation",
        "builds_on": {
            "RQ13": "results/frontier/diverse_hallucination_detector/ (PR #904)",
            "RQ16": "results/frontier/corrected_router_simulation/ (PR #912)",
            "RQ25": "results/frontier/out_of_sample_router/ (PR #929)",
            "RQ44": "results/frontier/bootstrap_threshold_stability/ (PR #963)",
        },
        "source_data": str(SRC_JSON.relative_to(PROJECT_ROOT)),
        "source_label": "external/sanity-check",
        "method": (
            "reanalysis only (no Whisper / no ASR run); for each n in "
            "{77, 154, 308, 616, 1232}, draw B=2000 bootstrap resamples of size n "
            "(with replacement from the 77 AISHELL-4 windows; duplicates allowed). "
            "On each resample: calibrate the lang-id entropy threshold maximising "
            "sensitivity at >= 90% specificity (RQ13/RQ16/RQ25/RQ44 rule). Record "
            "threshold; compute OOB cpWER on windows NOT drawn. Report threshold "
            "distribution (modality, percentile interval width) and OOB cpWER per n."
        ),
        "meeting_id": data["meeting_id"],
        "n_population_windows": n_pop,
        "n_hallucinated": n_hall,
        "n_clean": n_clean,
        "hallucination_label_rule": "always_separated_cpwer > 1.0",
        "routing_rule": (
            "lang_id_entropy >= threshold -> route MIXED (always_mixed_cpwer); "
            "else route SEPARATED (always_separated_cpwer). HIGH lang-id entropy = "
            "diverse multilingual gibberish = hallucination (RQ13/RQ16/RQ25/RQ44 convention)."
        ),
        "calibration_rule": (
            "sweep threshold 0.0-2.0 bits in 0.01 steps; select threshold maximising "
            "sensitivity at >= 90% specificity. Tie-breaker: higher specificity, then "
            "lower threshold. Verbatim from RQ44."
        ),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "sample_sizes": SAMPLE_SIZES,
            "mode_min_fraction": MODE_MIN_FRACTION,
            "seed_rule": (
                "n=77 uses seed 42 (directly comparable to RQ44); larger n use "
                "seed 42+n_sample so resamples for different n are independent."
            ),
        },
        "in_sample_reproduction": {
            "RQ25_threshold": RQ25_IN_SAMPLE_THRESHOLD,
            "RQ44_bootstrap_median_threshold": RQ44_BOOTSTRAP_MEDIAN_THRESHOLD,
            "threshold": in_sample["threshold"],
            "sensitivity": in_sample["sensitivity"],
            "specificity": in_sample["specificity"],
            "cpwer": in_sample_cpwer,
            "tp": in_sample["tp"], "fp": in_sample["fp"],
            "tn": in_sample["tn"], "fn": in_sample["fn"],
            "note": "Calibrated and evaluated on all 77 windows (in-sample). Reproduces RQ25/RQ44.",
        },
        "hypothesis_anchors": {
            "anchor_n": H45_ANCHOR_N,
            "H45a_max_modes": H45A_MAX_MODES,
            "H45b_max_width": H45B_MAX_WIDTH,
            "H45c_max_cpwer": H45C_MAX_CPWER,
        },
        "per_n": [
            {
                "n_sample": r["n_sample"],
                "expected_oob_size": round(r["expected_oob_size"], 6),
                "threshold_distribution": {
                    "median": round(r["threshold_distribution"]["median"], 6),
                    "mean": round(r["threshold_distribution"]["mean"], 6),
                    "std": round(r["threshold_distribution"]["std"], 6),
                    "min": round(r["threshold_distribution"]["min"], 6),
                    "max": round(r["threshold_distribution"]["max"], 6),
                    "percentile_2_5": round(r["threshold_distribution"]["percentile_2_5"], 6),
                    "percentile_97_5": round(r["threshold_distribution"]["percentile_97_5"], 6),
                    "interval_width": round(r["threshold_distribution"]["interval_width"], 6),
                    "n_modes": r["threshold_distribution"]["n_modes"],
                    "n_unique": r["threshold_distribution"]["n_unique"],
                    "dominant": round(r["threshold_distribution"]["dominant"], 6),
                    "dominant_fraction": round(r["threshold_distribution"]["dominant_fraction"], 6),
                    "modes": [
                        {"threshold": round(m["threshold"], 6),
                         "count": m["count"],
                         "fraction": round(m["fraction"], 6)}
                        for m in r["threshold_distribution"]["modes"]
                    ],
                },
                "oob_cpwer_distribution": {
                    "n_valid_resamples": r["oob_cpwer_distribution"]["n_valid_resamples"],
                    "mean_oob_size": round(r["oob_cpwer_distribution"]["mean_oob_size"], 6),
                    "median": round(r["oob_cpwer_distribution"]["median"], 6)
                              if not math.isnan(r["oob_cpwer_distribution"]["median"]) else None,
                    "mean": round(r["oob_cpwer_distribution"]["mean"], 6)
                            if not math.isnan(r["oob_cpwer_distribution"]["mean"]) else None,
                    "min": round(r["oob_cpwer_distribution"]["min"], 6)
                           if not math.isnan(r["oob_cpwer_distribution"]["min"]) else None,
                    "max": round(r["oob_cpwer_distribution"]["max"], 6)
                           if not math.isnan(r["oob_cpwer_distribution"]["max"]) else None,
                    "percentile_2_5": round(r["oob_cpwer_distribution"]["percentile_2_5"], 6)
                                      if not math.isnan(r["oob_cpwer_distribution"]["percentile_2_5"]) else None,
                    "percentile_97_5": round(r["oob_cpwer_distribution"]["percentile_97_5"], 6)
                                       if not math.isnan(r["oob_cpwer_distribution"]["percentile_97_5"]) else None,
                    "iqr": [
                        round(r["oob_cpwer_distribution"]["iqr"][0], 6)
                        if not math.isnan(r["oob_cpwer_distribution"]["iqr"][0]) else None,
                        round(r["oob_cpwer_distribution"]["iqr"][1], 6)
                        if not math.isnan(r["oob_cpwer_distribution"]["iqr"][1]) else None,
                    ],
                    "frac_below_1_10": round(r["oob_cpwer_distribution"]["frac_below_1_10"], 6)
                                       if not math.isnan(r["oob_cpwer_distribution"]["frac_below_1_10"]) else None,
                },
            }
            for r in per_n
        ],
        "hypothesis_verdicts": {
            "H45a": {
                "statement": (
                    f"At n={H45_ANCHOR_N}, threshold distribution is unimodal "
                    f"(<= {H45A_MAX_MODES} distinct thresholds with >=5% frequency)"
                ),
                "n_modes_at_anchor": anchor_thr["n_modes"],
                "modes_at_anchor": [
                    {"threshold": round(m["threshold"], 6),
                     "count": m["count"],
                     "fraction": round(m["fraction"], 6)}
                    for m in anchor_thr["modes"]
                ],
                "kill_threshold": f"> {H45A_MAX_MODES} modes with >=5% frequency",
                "supported": bool(h45a_supported),
            },
            "H45b": {
                "statement": (
                    f"At n={H45_ANCHOR_N}, 2.5/97.5 percentile interval width < {H45B_MAX_WIDTH}"
                ),
                "interval": [
                    round(anchor_thr["percentile_2_5"], 6),
                    round(anchor_thr["percentile_97_5"], 6),
                ],
                "width": round(anchor_thr["interval_width"], 6),
                "kill_threshold": f">= {H45B_MAX_WIDTH}",
                "supported": bool(h45b_supported),
            },
            "H45c": {
                "statement": (
                    f"At n={H45_ANCHOR_N}, median OOB cpWER < {H45C_MAX_CPWER}"
                ),
                "median_oob_cpwer": round(anchor_oob["median"], 6)
                                    if not math.isnan(anchor_oob["median"]) else None,
                "n_valid_resamples": anchor_oob["n_valid_resamples"],
                "kill_threshold": f">= {H45C_MAX_CPWER}",
                "supported": bool(h45c_supported),
                "caveat": (
                    "At n=616 the expected OOB size is ~0.025 windows; most "
                    "resamples have an empty OOB set. The median is computed over "
                    "the few resamples with non-empty OOB, so it is noisy and "
                    "should be read as illustrative, not as a stable estimate."
                ),
            },
        },
    }

    # ----------------------------------------------------------- per-n CSV (summary)
    csv_fields = [
        "n_sample", "expected_oob_size", "mean_oob_size",
        "thr_median", "thr_mean", "thr_std", "thr_min", "thr_max",
        "thr_p2_5", "thr_p97_5", "thr_interval_width",
        "thr_n_modes", "thr_n_unique", "thr_dominant", "thr_dominant_fraction",
        "oob_n_valid", "oob_median", "oob_mean", "oob_p2_5", "oob_p97_5",
        "oob_min", "oob_max", "oob_frac_below_1_10",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields)
        wr.writeheader()
        for r in per_n:
            td = r["threshold_distribution"]
            od = r["oob_cpwer_distribution"]
            wr.writerow({
                "n_sample": r["n_sample"],
                "expected_oob_size": round(r["expected_oob_size"], 6),
                "mean_oob_size": round(od["mean_oob_size"], 6),
                "thr_median": round(td["median"], 6),
                "thr_mean": round(td["mean"], 6),
                "thr_std": round(td["std"], 6),
                "thr_min": round(td["min"], 6),
                "thr_max": round(td["max"], 6),
                "thr_p2_5": round(td["percentile_2_5"], 6),
                "thr_p97_5": round(td["percentile_97_5"], 6),
                "thr_interval_width": round(td["interval_width"], 6),
                "thr_n_modes": td["n_modes"],
                "thr_n_unique": td["n_unique"],
                "thr_dominant": round(td["dominant"], 6),
                "thr_dominant_fraction": round(td["dominant_fraction"], 6),
                "oob_n_valid": od["n_valid_resamples"],
                "oob_median": "" if math.isnan(od["median"]) else round(od["median"], 6),
                "oob_mean": "" if math.isnan(od["mean"]) else round(od["mean"], 6),
                "oob_p2_5": "" if math.isnan(od["percentile_2_5"]) else round(od["percentile_2_5"], 6),
                "oob_p97_5": "" if math.isnan(od["percentile_97_5"]) else round(od["percentile_97_5"], 6),
                "oob_min": "" if math.isnan(od["min"]) else round(od["min"], 6),
                "oob_max": "" if math.isnan(od["max"]) else round(od["max"], 6),
                "oob_frac_below_1_10": "" if math.isnan(od["frac_below_1_10"]) else round(od["frac_below_1_10"], 6),
            })

    # ----------------------------------------------------------- write JSON (full, with per-bootstrap arrays)
    full = dict(summary)
    full["per_bootstrap_arrays"] = {
        f"n_{r['n_sample']}": {
            "thresholds": [round(float(t), 6) for t in r["per_bootstrap"]["thresholds"]],
            "oob_cpwer": [round(float(c), 6) if not math.isnan(float(c)) else None
                          for c in r["per_bootstrap"]["oob_cpwer"]],
            "n_oob": [int(x) for x in r["per_bootstrap"]["n_oob"]],
        }
        for r in per_n
    }
    OUT_JSON.write_text(
        json.dumps(full, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------- console
    print(f"=== RQ45: Multi-meeting threshold stability simulation ===")
    print(f"Label: experimental/frontier  |  Source: {SRC_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Population: {n_pop} AISHELL-4 windows ({n_hall} hall / {n_clean} clean)")
    print(f"Bootstrap: B={N_BOOT}, seed={SEED}, sample_sizes={SAMPLE_SIZES}")
    print()
    print("In-sample reproduction (calibrate + evaluate on all 77, RQ25/RQ44 reference):")
    print(f"  threshold         : {in_sample['threshold']:.4f}  (RQ25/RQ44 reported 0.38)")
    print(f"  sensitivity       : {in_sample['sensitivity']:.4f}")
    print(f"  specificity       : {in_sample['specificity']:.4f}")
    print(f"  corrected cpWER   : {in_sample_cpwer:.6f}")
    print()
    print(f"{'n':>6} | {'thr_med':>8} {'thr_wid':>8} {'n_modes':>8} {'dom':>6} {'dom%':>6}"
          f" | {'oob_n_val':>9} {'oob_med':>8} {'oob%<1.1':>8}")
    print("-" * 90)
    for r in per_n:
        td = r["threshold_distribution"]
        od = r["oob_cpwer_distribution"]
        oob_med_str = f"{od['median']:.4f}" if not math.isnan(od["median"]) else "nan"
        oob_frac_str = (f"{od['frac_below_1_10']:.3f}"
                        if not math.isnan(od["frac_below_1_10"]) else "nan")
        print(f"{r['n_sample']:>6} | {td['median']:>8.4f} {td['interval_width']:>8.4f} "
              f"{td['n_modes']:>8d} {td['dominant']:>6.2f} {td['dominant_fraction']:>6.3f}"
              f" | {od['n_valid_resamples']:>9d} {oob_med_str:>8} {oob_frac_str:>8}")
    print()
    print(f"Hypothesis verdicts (anchored at n={H45_ANCHOR_N}):")
    print(f"  H45a (<= {H45A_MAX_MODES} modes with >=5%):     "
          f"{'SUPPORTED' if h45a_supported else 'KILLED'}  "
          f"(n_modes={anchor_thr['n_modes']})")
    print(f"      modes at n={H45_ANCHOR_N}:")
    for m in anchor_thr["modes"]:
        print(f"        threshold={m['threshold']:.4f}  count={m['count']}  frac={m['fraction']:.3f}")
    print(f"  H45b (pct interval width < {H45B_MAX_WIDTH}):     "
          f"{'SUPPORTED' if h45b_supported else 'KILLED'}  "
          f"(width={anchor_thr['interval_width']:.4f}, "
          f"interval=[{anchor_thr['percentile_2_5']:.4f}, {anchor_thr['percentile_97_5']:.4f}])")
    oob_med_str = (f"{anchor_oob['median']:.4f}"
                   if not math.isnan(anchor_oob["median"]) else "nan")
    print(f"  H45c (median OOB cpWER < {H45C_MAX_CPWER}):    "
          f"{'SUPPORTED' if h45c_supported else 'KILLED'}  "
          f"(median OOB cpWER={oob_med_str}, "
          f"n_valid={anchor_oob['n_valid_resamples']}/{N_BOOT})")
    print()
    print(f"Wrote: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
