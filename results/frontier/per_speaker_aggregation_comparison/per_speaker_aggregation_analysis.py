"""RQ56: Per-speaker lang-id entropy aggregation comparison.

REANALYSIS ONLY -- no Whisper / no ASR / no LLM / no ollama is run. This script
reads the existing AISHELL-4 external-validation results
(``results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json``,
label ``external/sanity-check``, PR #890) and compares four ways of aggregating
the per-speaker language-id entropy across the separated speaker tracks into one
window-level hallucination score.

Motivation (RQ13 / RQ16 / RQ25 / RQ37 / RQ44)
---------------------------------------------
RQ13/RQ16/RQ25/RQ44 all aggregate per-speaker lang-id entropy by **MAX** (the
worst-case speaker track). RQ44 (PR #959) showed this MAX threshold is
6-modal on n=77 (width 0.94). RQ37 (PR #946) showed the worst speaker
contributes 96.5% of cpWER in the top-10 windows -- the worst-case speaker
dominates the catastrophic outcome. RQ56 asks: **does the aggregation function
(MAX vs SUM vs MEAN vs MIN) change the corrected router's detection performance
and corrected cpWER?**

MAX is the worst-case convention. If the worst speaker dominates cpWER, MAX may
already be the optimal aggregation. Alternatively, SUM or MEAN may smooth
single-speaker noise and produce a more stable (fewer-mode) threshold
distribution.

Method
------
For each of the 77 windows compute ``language_id_entropy`` for each separated
speaker track (RQ13/RQ16/RQ25/RQ44 verbatim primitive). Aggregate the
per-speaker entropies into one window-level score using four functions:

  - **MAX**  -- max over non-empty speaker tracks (RQ13/RQ16/RQ25/RQ44 convention)
  - **SUM**  -- sum over non-empty speaker tracks (scales with speaker count)
  - **MEAN** -- mean over non-empty speaker tracks (smooths speaker count)
  - **MIN**  -- min over non-empty speaker tracks (best-case speaker)

For each aggregation: calibrate the threshold maximising sensitivity at
>= 90% specificity (RQ25/RQ44 rule); compute the in-sample corrected cpWER
(route MIXED when score >= threshold else SEPARATED); bootstrap the threshold
distribution (B=10000, seed=42, resample size 77 with replacement) using the
RQ44 framework (calibrate on in-bag, evaluate OOB cpWER on the held-out
windows at the resample's threshold).

Threshold grid: per-aggregation adaptive so SUM (which scales with speaker
count) is not penalised by a MAX-sized grid. MAX uses the RQ44-exact 0.00-2.00
grid (201 points, step 0.01) so it reproduces RQ44's 0.38 threshold; the other
three use 0.00 to ceil(max_observed * 1.05), step 0.01, computed ONCE from the
full-sample scores and reused for all bootstrap resamples (the candidate set
must be fixed across resamples for the threshold distribution to be
comparable).

Pre-registered hypotheses (issue #977)
--------------------------------------
- H56a: MAX aggregation achieves the highest sensitivity at >= 90% specificity.
        Killed if any other aggregation achieves strictly higher sensitivity.
- H56b: SUM aggregation produces fewer bootstrap threshold modes than MAX
        (smoother distribution). Killed if SUM's mode count (>= 5% frequency,
        RQ48's ``count_modes``) >= MAX's mode count. n_unique (all distinct
        thresholds) is reported secondarily for traceability.
- H56c: All 4 aggregations achieve corrected cpWER <= 1.10 (aggregation choice
        doesn't break deployability). Killed if any aggregation's in-sample
        corrected cpWER > 1.10.

Routing rule (RQ13/RQ16/RQ25/RQ44 convention)
---------------------------------------------
HIGH lang-id entropy = diverse multilingual gibberish = hallucination. The
detector flags the separated track when ``aggregated_score >= threshold``:

    if aggregated_score >= threshold -> route MIXED  (always_mixed_cpwer)
    else                             -> route SEPARATED (always_separated_cpwer)

The corrected router's per-window cpWER is the chosen route's stored cpWER.
Per the project's hard safety rules, cpWER / references are NOT used as routing
input -- only as calibration and OOB evaluation labels.

This script is pure reanalysis (numpy + stdlib only; scipy / sklearn / Whisper
/ meeteval / LLM are NOT required). The detector primitives
(``script_category``, ``language_id_entropy``), the calibration rule
(``calibrate_threshold_at_spec``), the bootstrap framework
(``bootstrap_indices``, ``out_of_bag_cpwer``), and the distribution summary
(``percentile_interval``, ``threshold_distribution``) are lifted verbatim from
RQ44 so the MAX arm reproduces RQ44 exactly.

Label: experimental/frontier. Closes #977. Builds on RQ13 (PR #904), RQ16
(PR #912), RQ25 (PR #929), RQ37 (PR #946), and RQ44 (PR #959).
"""
from __future__ import annotations

import csv
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Callable

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
OUT_DIR = PROJECT_ROOT / "results" / "frontier" / "per_speaker_aggregation_comparison"
OUT_CSV = OUT_DIR / "per_speaker_aggregation_results.csv"
OUT_JSON = OUT_DIR / "per_speaker_aggregation_results.json"

# ------------------------------------------------------------------ constants
CATASTROPHIC_CPWER = 1.0    # cpWER > 1.0 => insertions dominate (hallucination label)
TARGET_SPECIFICITY = 0.90   # calibrate each aggregation to >= 90% specificity
N_BOOT = 10000
SEED = 42
EPS = 1e-9

# The four aggregations compared. Order is fixed so output tables are stable.
AGGREGATIONS: tuple[str, ...] = ("max", "sum", "mean", "min")

# MAX uses the RQ44-exact grid so the MAX arm reproduces RQ44's 0.38 threshold
# (201 candidates, 0.00-2.00 bits, step 0.01). The other three aggregations use
# an adaptive grid built from their full-sample observed range (see
# ``build_adaptive_grid``).
MAX_GRID: list[float] = [round(0.01 * i, 2) for i in range(0, 201)]

# RQ44 references (for the MAX-arm reproduction check).
RQ44_IN_SAMPLE_THRESHOLD = 0.38
RQ44_IN_SAMPLE_CPWER = 1.043
RQ44_N_UNIQUE_MODES = 6

# Hypothesis kill thresholds.
H56C_MAX_CPWER = 1.10

# "Mode" definition for H56b (matches RQ48's count_modes): a distinct threshold
# value whose bootstrap frequency is >= MIN_MODE_FRACTION. This is the explicit
# kill-condition metric the task METHOD specifies (RQ48's count_modes), reported
# alongside RQ44's n_unique (all distinct thresholds) and modes_within_10pct for
# traceability.
MIN_MODE_FRACTION = 0.05


# ------------------------------------------------------------- script detection
def script_category(ch: str) -> str:
    """Map a character to a coarse Unicode script category (RQ13/RQ16/RQ25/RQ44
    verbatim).

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
    """Shannon entropy (bits) over the script-category distribution of the text
    (RQ13/RQ16/RQ25/RQ44 verbatim).

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


# ----------------------------------------------------------- per-speaker + aggregate
def per_speaker_entropies(window: dict[str, Any]) -> list[float]:
    """Return ``language_id_entropy`` for each NON-EMPTY separated speaker track
    in the window.

    None / whitespace-only speaker texts are skipped (they never carry a
    hallucination signal; this matches RQ13/RQ44's ``max_across_speakers``
    filtering). Returns ``[]`` if every speaker track is empty."""
    out: list[float] = []
    for t in window.get("separated_text_per_speaker", {}).values():
        if t is None:
            continue
        s = str(t)
        if not s.strip():
            continue
        out.append(language_id_entropy(s))
    return out


_AGG_FUNCS: dict[str, Callable[[list[float]], float]] = {
    "max": lambda xs: float(max(xs)) if xs else 0.0,
    "sum": lambda xs: float(sum(xs)) if xs else 0.0,
    "mean": lambda xs: float(sum(xs) / len(xs)) if xs else 0.0,
    "min": lambda xs: float(min(xs)) if xs else 0.0,
}


def aggregate_scores(per_speaker_vals: list[float], agg: str) -> float:
    """Aggregate a list of per-speaker entropy values into one window-level
    score using one of ``AGGREGATIONS`` (max / sum / mean / min).

    Returns 0.0 for an empty list (window with no non-empty speaker track).
    Raises ``ValueError`` for an unknown aggregation name."""
    if agg not in _AGG_FUNCS:
        raise ValueError(f"unknown aggregation: {agg!r} (expected one of {AGGREGATIONS})")
    return _AGG_FUNCS[agg](per_speaker_vals)


def aggregate_window(window: dict[str, Any], agg: str) -> float:
    """Convenience: per-speaker entropies for ``window`` aggregated by ``agg``."""
    return aggregate_scores(per_speaker_entropies(window), agg)


# --------------------------------------------------------------- threshold grid
def build_adaptive_grid(scores: np.ndarray, margin: float = 1.05) -> list[float]:
    """Build a per-aggregation threshold grid from the observed score range.

    Grid is ``[0.00, 0.01, ..., ceil(max(scores) * margin)]`` with a 0.01 step.
    Returns ``[0.00]`` for empty/all-zero scores. The grid is computed ONCE
    from the full-sample scores and reused for every bootstrap resample so the
    candidate-threshold set is fixed across resamples (otherwise the threshold
    distribution would be incomparable across resamples). MAX does NOT use this
    -- it uses the fixed ``MAX_GRID`` (0.00-2.00) to reproduce RQ44."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return [0.00]
    hi = float(arr.max())
    if hi <= 0.0:
        return [0.00]
    # ``top`` is the integer ceiling of ``hi * margin`` (value units). The grid
    # runs from 0.00 to ``top`` in 0.01 steps, i.e. ``top * 100 + 1`` points.
    top = int(math.ceil(hi * margin))
    if top < 1:
        top = 1
    return [round(0.01 * i, 2) for i in range(0, top * 100 + 1)]


def grid_for(agg: str, scores: np.ndarray) -> list[float]:
    """Return the threshold grid for ``agg`` given the full-sample ``scores``.

    MAX -> ``MAX_GRID`` (RQ44-exact, 0.00-2.00). Others -> adaptive grid from
    the observed range. This keeps the MAX arm directly comparable to RQ44
    while giving SUM / MEAN / MIN a grid that covers their (larger for SUM,
    comparable for MEAN/MIN) observed range."""
    if agg == "max":
        return list(MAX_GRID)
    return build_adaptive_grid(scores)


# --------------------------------------------------------- threshold calibration
def calibrate_threshold_at_spec(
    scores: np.ndarray,
    labels: np.ndarray,
    grid: list[float] | None = None,
    target_spec: float = TARGET_SPECIFICITY,
) -> dict[str, Any]:
    """Sweep threshold over ``grid``; select the threshold with specificity
    >= ``target_spec`` and maximal sensitivity. Tie-breaker: higher specificity,
    then lower threshold (more sensitive).

    Convention (RQ13/RQ16/RQ25/RQ44): ``score >= threshold`` flags the window
    as hallucinated. Sensitivity = TP / (TP + FN); specificity = TN / (TN + FP).

    Returns the chosen threshold plus its sensitivity, specificity, and
    confusion counts. Faithful copy of RQ44's ``calibrate_threshold_at_spec``
    so the MAX arm reproduces RQ44's 0.38 threshold exactly."""
    if grid is None:
        grid = list(MAX_GRID)
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


def corrected_cpwer(
    scores: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    threshold: float,
) -> float:
    """Mean corrected-router cpWER at ``threshold`` over the supplied windows.

    Routing: ``score >= threshold`` -> route MIXED (``mixed_cpwer``); else
    SEPARATED (``sep_cpwer``). Returns ``nan`` for empty input."""
    scores = np.asarray(scores, dtype=float)
    mixed_cpwer = np.asarray(mixed_cpwer, dtype=float)
    sep_cpwer = np.asarray(sep_cpwer, dtype=float)
    if scores.size == 0:
        return float("nan")
    flagged = scores >= threshold - EPS
    selected = np.where(flagged, mixed_cpwer, sep_cpwer)
    return float(selected.mean())


# --------------------------------------------------------------------- bootstrap
def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Return an ``(n_boot, n)`` int array of bootstrap resample indices.

    Each row is an independent bootstrap resample: ``n`` indices drawn with
    replacement from ``{0, ..., n-1}``. Deterministic for a given seed.
    (RQ44 verbatim.)"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def out_of_bag_cpwer(
    scores: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    threshold: float,
    in_bag_idx: np.ndarray,
) -> dict[str, Any]:
    """Compute the corrected-router cpWER on the out-of-bag (OOB) windows --
    those NOT present in the bootstrap resample ``in_bag_idx`` -- at the given
    ``threshold``. (RQ44 verbatim.)

    Returns a dict with the mean selected cpWER over the OOB windows (``nan``
    if there are none), the OOB size, and the mixed/separated decision counts."""
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


def percentile_interval(
    values: np.ndarray, lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    """Return the ``(lo, hi)`` percentile interval of ``values`` using numpy's
    default linear interpolation. Returns ``(nan, nan)`` for empty input.
    (RQ44 verbatim.)"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def threshold_distribution(thresholds: np.ndarray, top_k: int = 5) -> dict[str, Any]:
    """Summarise the bootstrap threshold distribution: the modal value, its
    count and fraction, the number of distinct thresholds, the top-``top_k``
    most frequent values, and the "modes" -- values whose count is within 10%
    of the maximum count (to expose bimodality). (RQ44 verbatim.)

    The distinct-threshold count (``n_unique``) is the primary "mode count"
    used by H56b: it is the number of distinct operating points the
    calibration rule produces across the B resamples."""
    arr = np.asarray(thresholds, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mode": float("nan"), "mode_count": 0, "mode_fraction": 0.0,
                "n_unique": 0, "modes_within_10pct": [], "top_values": []}
    uniq, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)  # descending by count
    uniq = uniq[order]
    counts = counts[order]
    max_count = int(counts[0])
    modes = [
        (float(u), int(c), float(c / n))
        for u, c in zip(uniq, counts)
        if c >= max_count * 0.90
    ]
    top = [
        (float(u), int(c), float(c / n))
        for u, c in zip(uniq[:top_k], counts[:top_k])
    ]
    return {
        "mode": float(uniq[0]),
        "mode_count": max_count,
        "mode_fraction": float(max_count / n),
        "n_unique": int(uniq.size),
        "modes_within_10pct": modes,
        "top_values": top,
    }


def count_modes(
    thresholds: np.ndarray, min_fraction: float = MIN_MODE_FRACTION
) -> dict[str, Any]:
    """Count distinct threshold values whose bootstrap frequency is >=
    ``min_fraction`` (default 5%).

    Returns the mode count, the list of modes (threshold / count / fraction)
    sorted by descending count then ascending threshold for determinism, and the
    total number of distinct thresholds. This is RQ48's ``count_modes`` verbatim
    (``results/frontier/calibration_rule_comparison/calibration_rule_analysis.py``)
    and is the explicit kill-condition definition of "mode" for H56b: the task
    METHOD specifies counting threshold modes (>= 5% frequency) using RQ48's
    ``count_modes``."""
    arr = np.asarray(thresholds, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n_modes": 0, "modes": [], "n_unique": 0,
                "min_fraction": float(min_fraction)}
    uniq, counts = np.unique(arr, return_counts=True)
    fracs = counts / n
    mask = fracs >= min_fraction - EPS
    mode_idx = np.where(mask)[0]
    # Sort by descending count; stable sort keeps ascending-threshold order ties.
    order = mode_idx[np.argsort(-counts[mode_idx], kind="stable")]
    modes = [
        {"threshold": float(uniq[i]),
         "count": int(counts[i]),
         "fraction": float(fracs[i])}
        for i in order
    ]
    return {"n_modes": len(modes), "modes": modes,
            "n_unique": int(uniq.size), "min_fraction": float(min_fraction)}


# ------------------------------------------------------------- per-aggregation arm
def run_aggregation_arm(
    agg: str,
    scores: np.ndarray,
    labels: np.ndarray,
    mixed_cpwer: np.ndarray,
    sep_cpwer: np.ndarray,
    grid: list[float],
    boot_idx: np.ndarray,
) -> dict[str, Any]:
    """Run the full per-aggregation analysis: in-sample calibration + cpWER,
    then B bootstrap resamples (threshold + OOB cpWER per resample), then
    aggregate the threshold and OOB-cpWER distributions.

    ``boot_idx`` is the shared ``(N_BOOT, n)`` resample-index array (the SAME
    resamples are used for all four aggregations, so the comparison isolates
    the aggregation function and does not conflate it with resample-draw
    noise)."""
    n = len(scores)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    # In-sample calibration + corrected cpWER.
    in_sample = calibrate_threshold_at_spec(scores, labels, grid=grid)
    in_cpwer = corrected_cpwer(scores, mixed_cpwer, sep_cpwer, in_sample["threshold"])

    # Bootstrap: shared resamples, per-aggregation threshold + OOB cpWER.
    n_boot = boot_idx.shape[0]
    boot_thresholds = np.empty(n_boot, dtype=float)
    boot_oob_cpwer = np.empty(n_boot, dtype=float)
    boot_n_oob = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        idx = boot_idx[b]
        cal = calibrate_threshold_at_spec(scores[idx], labels[idx], grid=grid)
        thr = cal["threshold"]
        boot_thresholds[b] = thr
        oob = out_of_bag_cpwer(scores, mixed_cpwer, sep_cpwer, thr, idx)
        boot_oob_cpwer[b] = oob["cpwer"]
        boot_n_oob[b] = oob["n_oob"]

    # Threshold distribution.
    thr_median = float(np.median(boot_thresholds))
    thr_mean = float(np.mean(boot_thresholds))
    thr_std = float(np.std(boot_thresholds))
    thr_lo, thr_hi = percentile_interval(boot_thresholds, 2.5, 97.5)
    thr_min = float(np.min(boot_thresholds))
    thr_max = float(np.max(boot_thresholds))
    thr_width = thr_hi - thr_lo
    thr_dist = threshold_distribution(boot_thresholds)
    # RQ48's count_modes (>= 5% frequency) -- the explicit H56b kill-condition
    # metric the task METHOD specifies. Reported alongside n_unique (all distinct
    # thresholds) and modes_within_10pct (RQ44's 10%-of-max definition).
    modes_5pct = count_modes(boot_thresholds, MIN_MODE_FRACTION)

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
        oob_frac_below_110 = float(np.mean(valid_oob < H56C_MAX_CPWER))
    else:
        oob_median = oob_mean = float("nan")
        oob_lo = oob_hi = oob_min = oob_max = float("nan")
        oob_p25 = oob_p75 = float("nan")
        oob_frac_below_110 = float("nan")
    oob_n_oob_mean = float(np.mean(boot_n_oob[valid_mask])) if n_valid > 0 else 0.0

    return {
        "aggregation": agg,
        "grid": {"lo": float(grid[0]), "hi": float(grid[-1]),
                 "n_points": len(grid), "strategy": "max_fixed_rq44" if agg == "max"
                 else "adaptive"},
        "in_sample": {
            "threshold": round(in_sample["threshold"], 6),
            "sensitivity": round(in_sample["sensitivity"], 6),
            "specificity": round(in_sample["specificity"], 6),
            "cpwer": round(in_cpwer, 6),
            "tp": in_sample["tp"], "fp": in_sample["fp"],
            "tn": in_sample["tn"], "fn": in_sample["fn"],
            "n_pos": n_pos, "n_neg": n_neg,
        },
        "threshold_distribution": {
            "median": round(thr_median, 6),
            "mean": round(thr_mean, 6),
            "std": round(thr_std, 6),
            "min": round(thr_min, 6),
            "max": round(thr_max, 6),
            "percentile_2_5": round(thr_lo, 6),
            "percentile_97_5": round(thr_hi, 6),
            "interval_width": round(thr_width, 6),
            "mode": round(thr_dist["mode"], 6),
            "mode_count": thr_dist["mode_count"],
            "mode_fraction": round(thr_dist["mode_fraction"], 6),
            "n_unique": thr_dist["n_unique"],
            "n_modes_5pct": modes_5pct["n_modes"],
            "modes_5pct": [
                {"threshold": round(m["threshold"], 6),
                 "count": m["count"], "fraction": round(m["fraction"], 6)}
                for m in modes_5pct["modes"]
            ],
            "min_mode_fraction": float(MIN_MODE_FRACTION),
            "modes_within_10pct": [
                {"threshold": round(t, 6), "count": c, "fraction": round(f, 6)}
                for t, c, f in thr_dist["modes_within_10pct"]
            ],
            "top_values": [
                {"threshold": round(t, 6), "count": c, "fraction": round(f, 6)}
                for t, c, f in thr_dist["top_values"]
            ],
        },
        "oob_cpwer_distribution": {
            "n_valid_resamples": n_valid,
            "mean_oob_size": round(oob_n_oob_mean, 4),
            "median": round(oob_median, 6) if not math.isnan(oob_median) else None,
            "mean": round(oob_mean, 6) if not math.isnan(oob_mean) else None,
            "min": round(oob_min, 6) if not math.isnan(oob_min) else None,
            "max": round(oob_max, 6) if not math.isnan(oob_max) else None,
            "percentile_2_5": round(oob_lo, 6) if not math.isnan(oob_lo) else None,
            "percentile_97_5": round(oob_hi, 6) if not math.isnan(oob_hi) else None,
            "iqr": [round(oob_p25, 6) if not math.isnan(oob_p25) else None,
                    round(oob_p75, 6) if not math.isnan(oob_p75) else None],
            "frac_below_1_10": round(oob_frac_below_110, 6)
            if not math.isnan(oob_frac_below_110) else None,
        },
        "per_bootstrap": {
            "thresholds": [round(float(t), 6) for t in boot_thresholds],
            "oob_cpwer": [round(float(c), 6) if not math.isnan(float(c)) else None
                          for c in boot_oob_cpwer],
            "n_oob": [int(x) for x in boot_n_oob],
        },
    }


# --------------------------------------------------------------------- driver
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    windows = data["windows"]
    n = len(windows)

    mixed_cpwer = np.array([float(w["always_mixed_cpwer"]) for w in windows], dtype=float)
    sep_cpwer = np.array([float(w["always_separated_cpwer"]) for w in windows], dtype=float)
    rv2_cpwer = np.array([float(w["router_v2_cpwer"]) for w in windows], dtype=float)
    oracle_cpwer = np.array([float(w["oracle_best_cpwer"]) for w in windows], dtype=float)
    labels = (sep_cpwer > CATASTROPHIC_CPWER).astype(int)
    n_hall = int(labels.sum())
    n_clean = int((labels == 0).sum())

    # Per-window per-aggregation scores + per-speaker entropy lists.
    per_window: list[dict[str, Any]] = []
    per_speaker_lists: list[list[float]] = []
    for w in windows:
        ents = per_speaker_entropies(w)
        per_speaker_lists.append(ents)
        row = {
            "window_id": w["window_id"],
            "overlap_label": w["overlap_label"],
            "num_speakers": w["num_speakers"],
            "n_nonempty_speakers": len(ents),
            "always_separated_cpwer": round(float(w["always_separated_cpwer"]), 6),
            "always_mixed_cpwer": round(float(w["always_mixed_cpwer"]), 6),
            "router_v2_cpwer": round(float(w["router_v2_cpwer"]), 6),
            "oracle_best_cpwer": round(float(w["oracle_best_cpwer"]), 6),
            "hallucinated": bool(float(w["always_separated_cpwer"]) > CATASTROPHIC_CPWER),
            "per_speaker_entropies": [round(e, 6) for e in ents],
        }
        for agg in AGGREGATIONS:
            row[f"score_{agg}"] = round(aggregate_scores(ents, agg), 6)
        per_window.append(row)

    # Shared bootstrap resamples (same draws for all 4 aggregations).
    boot_idx = bootstrap_indices(n, N_BOOT, SEED)

    # Per-aggregation arms.
    arms: dict[str, dict[str, Any]] = {}
    for agg in AGGREGATIONS:
        scores = np.array([aggregate_scores(pl, agg) for pl in per_speaker_lists],
                          dtype=float)
        grid = grid_for(agg, scores)
        arms[agg] = run_aggregation_arm(
            agg, scores, labels, mixed_cpwer, sep_cpwer, grid, boot_idx,
        )

    # --------------------------------------------------------- hypothesis verdicts
    sens = {a: arms[a]["in_sample"]["sensitivity"] for a in AGGREGATIONS}
    max_sens = sens["max"]
    other_best_sens = max(sens[a] for a in AGGREGATIONS if a != "max")
    # H56a: MAX achieves the HIGHEST sensitivity. Killed if any other
    # aggregation achieves STRICTLY higher sensitivity (by more than EPS).
    h56a_supported = max_sens >= other_best_sens - EPS

    n_modes = {a: arms[a]["threshold_distribution"]["n_modes_5pct"]
               for a in AGGREGATIONS}
    n_unique = {a: arms[a]["threshold_distribution"]["n_unique"] for a in AGGREGATIONS}
    # H56b: SUM produces FEWER bootstrap threshold modes than MAX. The task
    # METHOD specifies counting modes (>= 5% frequency) using RQ48's count_modes,
    # so the kill metric is n_modes_5pct (SUM's mode count >= MAX's => killed).
    # n_unique (all distinct thresholds) is reported secondarily for traceability.
    h56b_supported = n_modes["sum"] < n_modes["max"]

    in_cpwer = {a: arms[a]["in_sample"]["cpwer"] for a in AGGREGATIONS}
    # H56c: ALL 4 aggregations achieve corrected cpWER <= 1.10. Killed if any
    # aggregation's in-sample cpWER > 1.10.
    h56c_supported = all(in_cpwer[a] <= H56C_MAX_CPWER + EPS for a in AGGREGATIONS)

    # ----------------------------------------------------------- comparison table
    comparison = []
    for agg in AGGREGATIONS:
        a = arms[agg]
        comparison.append({
            "aggregation": agg,
            "threshold": a["in_sample"]["threshold"],
            "sensitivity": a["in_sample"]["sensitivity"],
            "specificity": a["in_sample"]["specificity"],
            "in_sample_cpwer": a["in_sample"]["cpwer"],
            "median_oob_cpwer": a["oob_cpwer_distribution"]["median"],
            "frac_oob_below_1_10": a["oob_cpwer_distribution"]["frac_below_1_10"],
            "bootstrap_threshold_median": a["threshold_distribution"]["median"],
            "bootstrap_interval_width": a["threshold_distribution"]["interval_width"],
            "n_unique_thresholds": a["threshold_distribution"]["n_unique"],
            "n_modes_5pct": a["threshold_distribution"]["n_modes_5pct"],
            "modes_within_10pct": len(a["threshold_distribution"]["modes_within_10pct"]),
        })

    summary: dict[str, Any] = {
        "label": "experimental/frontier",
        "rq": "RQ56: Per-speaker lang-id entropy aggregation comparison",
        "closes_issue": 977,
        "source_data": str(SRC_JSON.relative_to(PROJECT_ROOT)),
        "source_label": "external/sanity-check",
        "builds_on": {
            "RQ13": "results/frontier/diverse_hallucination_detector/ (PR #904)",
            "RQ16": "results/frontier/corrected_router_simulation/ (PR #912)",
            "RQ25": "results/frontier/out_of_sample_router/ (PR #929)",
            "RQ37": "PR #946 (worst speaker = 96.5% of cpWER in top-10 windows)",
            "RQ44": "results/frontier/bootstrap_threshold_stability/ (PR #959)",
        },
        "method": (
            "reanalysis only (no Whisper / no ASR / no LLM); per-speaker lang-id "
            "entropy aggregated by MAX / SUM / MEAN / MIN. For each aggregation: "
            "calibrate threshold (max sensitivity at >= 90% specificity), compute "
            "in-sample corrected cpWER, bootstrap threshold + OOB cpWER "
            "distribution (B=10000, seed=42, shared resamples across aggregations). "
            "MAX uses the RQ44-exact 0.00-2.00 grid; SUM/MEAN/MIN use an adaptive "
            "grid covering their observed range."
        ),
        "meeting_id": data["meeting_id"],
        "n_windows": n,
        "n_hallucinated": n_hall,
        "n_clean": n_clean,
        "hallucination_label_rule": "always_separated_cpwer > 1.0",
        "routing_rule": (
            "aggregated_score >= threshold -> route MIXED (always_mixed_cpwer); "
            "else route SEPARATED (always_separated_cpwer). HIGH lang-id entropy "
            "= diverse multilingual gibberish = hallucination "
            "(RQ13/RQ16/RQ25/RQ44 convention). cpWER / references are NOT used "
            "as routing input."
        ),
        "calibration_rule": (
            "sweep threshold over the per-aggregation grid in 0.01 steps; select "
            "threshold maximising sensitivity at >= 90% specificity. Tie-breaker: "
            "higher specificity, then lower threshold."
        ),
        "aggregations_compared": list(AGGREGATIONS),
        "aggregation_definitions": {
            "max": "max over non-empty per-speaker entropy values (worst-case speaker; RQ13/RQ16/RQ25/RQ44 convention)",
            "sum": "sum over non-empty per-speaker entropy values (scales with speaker count)",
            "mean": "mean over non-empty per-speaker entropy values (smooths speaker count)",
            "min": "min over non-empty per-speaker entropy values (best-case speaker)",
        },
        "threshold_grid_strategy": (
            "MAX -> fixed 0.00-2.00 grid (RQ44-exact, reproduces 0.38 threshold). "
            "SUM/MEAN/MIN -> adaptive grid 0.00 to ceil(max_observed*1.05), step "
            "0.01, computed ONCE from full-sample scores and reused for all "
            "bootstrap resamples so the candidate set is fixed across resamples."
        ),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "resample_size": n,
            "shared_resamples": (
                "the SAME (N_BOOT, n) resample-index array is used for all 4 "
                "aggregations, so the comparison isolates the aggregation function "
                "and does not conflate it with resample-draw noise"
            ),
            "expected_oob_size": round(n * ((1 - 1 / n) ** n), 4),
        },
        "rq44_reproduction_check": {
            "RQ44_threshold": RQ44_IN_SAMPLE_THRESHOLD,
            "RQ44_cpwer": RQ44_IN_SAMPLE_CPWER,
            "RQ44_n_unique_modes": RQ44_N_UNIQUE_MODES,
            "max_arm_threshold": arms["max"]["in_sample"]["threshold"],
            "max_arm_cpwer": arms["max"]["in_sample"]["cpwer"],
            "max_arm_n_unique": arms["max"]["threshold_distribution"]["n_unique"],
            "note": (
                "the MAX arm should reproduce RQ44's in-sample threshold (0.38) "
                "and cpwer (1.043) since it uses the same detector primitive, the "
                "same 0.00-2.00 grid, and the same calibration rule."
            ),
        },
        "comparison_table": comparison,
        "arms": arms,
        "hypothesis_verdicts": {
            "H56a": {
                "statement": "MAX aggregation achieves the highest sensitivity at >= 90% specificity",
                "max_sensitivity": round(max_sens, 6),
                "other_best_sensitivity": round(other_best_sens, 6),
                "per_aggregation_sensitivity": {a: round(sens[a], 6) for a in AGGREGATIONS},
                "kill_rule": "killed if any other aggregation achieves strictly higher sensitivity",
                "supported": bool(h56a_supported),
            },
            "H56b": {
                "statement": "SUM aggregation produces fewer bootstrap threshold modes than MAX (smoother distribution)",
                "max_n_modes_5pct": n_modes["max"],
                "sum_n_modes_5pct": n_modes["sum"],
                "per_aggregation_n_modes_5pct": {a: n_modes[a] for a in AGGREGATIONS},
                "max_n_unique": n_unique["max"],
                "sum_n_unique": n_unique["sum"],
                "per_aggregation_n_unique": {a: n_unique[a] for a in AGGREGATIONS},
                "kill_rule": ("killed if SUM's mode count (>= 5% frequency, RQ48 "
                              "count_modes) >= MAX's mode count"),
                "supported": bool(h56b_supported),
                "note": (
                    "Primary kill metric = n_modes_5pct (RQ48's count_modes: "
                    "distinct thresholds with >= 5% bootstrap frequency), as the "
                    "task METHOD specifies. n_unique (all distinct thresholds, "
                    "RQ44's '6-modal' usage) and modes_within_10pct (RQ44's "
                    "10%-of-max definition) are reported secondarily in each arm "
                    "for traceability."
                ),
            },
            "H56c": {
                "statement": "All 4 aggregations achieve corrected cpWER <= 1.10",
                "per_aggregation_in_sample_cpwer": {a: round(in_cpwer[a], 6) for a in AGGREGATIONS},
                "kill_threshold": H56C_MAX_CPWER,
                "kill_rule": "killed if any aggregation's in-sample corrected cpWER > 1.10",
                "supported": bool(h56c_supported),
            },
        },
    }

    # ----------------------------------------------------------- per-window CSV
    csv_fields = (
        ["window_id", "overlap_label", "num_speakers", "n_nonempty_speakers",
         "always_separated_cpwer", "always_mixed_cpwer", "router_v2_cpwer",
         "oracle_best_cpwer", "hallucinated"]
        + [f"score_{agg}" for agg in AGGREGATIONS]
    )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields)
        wr.writeheader()
        for r in per_window:
            wr.writerow({k: r.get(k, "") for k in csv_fields})

    # ----------------------------------------------------------- write JSON
    summary_with_windows = dict(summary)
    summary_with_windows["per_window_scores"] = per_window
    OUT_JSON.write_text(
        json.dumps(summary_with_windows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------- console
    print(f"=== RQ56: Per-speaker lang-id entropy aggregation comparison "
          f"(AISHELL-4, {n} windows) ===")
    print(f"Label: experimental/frontier  |  Source: {SRC_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Hallucination label: always_separated_cpwer > 1.0 -> {n_hall} hall / {n_clean} clean")
    print(f"Bootstrap: B={N_BOOT}, seed={SEED}, shared resamples across aggregations")
    print()
    hdr = (f"{'agg':5s} {'thresh':>9s} {'sens':>7s} {'spec':>7s} {'cpwer':>8s} "
           f"{'oob_med':>9s} {'oob<1.1':>8s} {'thr_med':>9s} {'thr_wid':>8s} "
           f"{'n_uniq':>7s}")
    print(hdr)
    for c in comparison:
        print(f"{c['aggregation']:5s} {c['threshold']:9.4f} "
              f"{c['sensitivity']:7.4f} {c['specificity']:7.4f} "
              f"{c['in_sample_cpwer']:8.4f} "
              f"{(c['median_oob_cpwer'] if c['median_oob_cpwer'] is not None else float('nan')):9.4f} "
              f"{(c['frac_oob_below_1_10'] if c['frac_oob_below_1_10'] is not None else float('nan')):8.4f} "
              f"{c['bootstrap_threshold_median']:9.4f} "
              f"{c['bootstrap_interval_width']:8.4f} "
              f"{c['n_unique_thresholds']:7d}")
    print()
    print("Hypothesis verdicts:")
    print(f"  H56a (MAX highest sensitivity):  "
          f"{'SUPPORTED' if h56a_supported else 'KILLED'}  "
          f"(MAX sens={max_sens:.4f}, other best={other_best_sens:.4f})")
    print(f"  H56b (SUM fewer modes than MAX): "
          f"{'SUPPORTED' if h56b_supported else 'KILLED'}  "
          f"(SUM n_modes5%={n_modes['sum']}, MAX n_modes5%={n_modes['max']}; "
          f"n_unique SUM={n_unique['sum']}, MAX={n_unique['max']})")
    print(f"  H56c (all 4 cpWER <= 1.10):       "
          f"{'SUPPORTED' if h56c_supported else 'KILLED'}  "
          f"(per-agg cpWER={ {a: round(in_cpwer[a],4) for a in AGGREGATIONS} })")
    print()
    print(f"RQ44 reproduction (MAX arm): threshold={arms['max']['in_sample']['threshold']:.4f} "
          f"(RQ44={RQ44_IN_SAMPLE_THRESHOLD}), cpwer={arms['max']['in_sample']['cpwer']:.4f} "
          f"(RQ44={RQ44_IN_SAMPLE_CPWER}), n_unique={arms['max']['threshold_distribution']['n_unique']} "
          f"(RQ44={RQ44_N_UNIQUE_MODES})")
    print(f"Wrote: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
