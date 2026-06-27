"""RQ66: Shrinkage + F1 combined threshold calibration.

REANALYSIS ONLY -- no Whisper / no ASR / no LLM. This script reuses RQ44's
bootstrap framework (PR #963, ``results/frontier/bootstrap_threshold_stability/``),
RQ48's F1 calibration rule + ``count_modes`` (PR #965,
``results/frontier/calibration_rule_comparison/``), RQ61's shrinkage mechanism
(PR #991, ``results/frontier/shrinkage_threshold_calibration/``), and the same
AISHELL-4 external-validation windows
(``results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json``,
label ``external/sanity-check``, PR #890) to ask: does combining a Bayesian
shrinkage prior with the F1 calibration rule achieve <= 2 modes?

Motivation (RQ44 -> RQ48 -> RQ61 -> RQ66)
-----------------------------------------
RQ44 showed the bootstrap threshold distribution is 5-modal (6 distinct values)
over [0.01, 0.95] under the "max sensitivity at >= 90% specificity" rule
(H44b KILLED, width 0.94). RQ48 decomposed the modality: the high-threshold
modes (0.87, 0.95) are rule artefacts that the smooth F1 rule eliminates (F1
gives 2 modes), but the low-threshold "Mode S catch" mode (0.01) persists under
every rule RQ48 tried -- RQ48 declared it "calibration-rule-invariant". RQ61
showed a Bayesian shrinkage prior (maximise ``sensitivity - lambda*|t - 0.38|``
at >= 90% specificity) ELIMINATES the 0.01 mode (overturning RQ48), but the two
high-threshold specificity-constraint modes (0.84, 0.87) survive (3 modes at
lambda=1.0). RQ61's FINDINGS explicitly predicted: "the natural next experiment
is shrinkage + a smooth rule (e.g. maximise ``F1 - lambda*|t - 0.38|``):
shrinkage kills the 0.01 mode, the smooth rule kills the 0.84/0.87 modes, and
the combination is predicted to reach <= 2 modes." RQ66 tests that prediction.

Method (RQ66)
-------------
The combined calibration rule maximises the shrinkage-penalised F1 objective

    objective(t) = F1(t) - lambda * |t - prior_mean|,   prior_mean = 0.38

over the same grid {0.00, 0.01, ..., 2.00} (201 points), where ``F1(t)`` is
RQ48's exact F1 score (2*prec*rec/(prec+rec)) and ``prior_mean = 0.38`` is RQ44's
bootstrap median (RQ61's prior). Tie-break: higher objective -> higher F1 ->
lower threshold (the last via the first True in the ascending grid, matching
RQ48's lowest-threshold convention). At ``lambda = 0`` the objective reduces to
pure F1 and the tie-break matches RQ48's ``calibrate_f1`` exactly -- this is
verified by a direct equivalence test. ``lambda`` in {0.0, 0.01, 0.1, 0.5, 1.0}
(RQ61's grid).

Bayesian framing. The L1 penalty ``-lambda*|t - 0.38|`` is the log of a Laplace
prior centred at 0.38, so the combined estimate is the MAP (posterior mode)
under a Laplace shrinkage prior with F1 as the (unnormalised) likelihood. This
is the faithful realisation of RQ61's predicted combination. The issue text
describes the prior loosely as "Beta(2,2)"; RQ61's own FINDINGS clarifies the
shrinkage is "a regularised point-estimate calibration toward a data-derived
centre" (0.38), not an external Beta(2,2) Bayes prior. To honour the issue's
literal "Beta(2,2) prior, posterior mode" phrasing, a secondary variant
(``calibrate_shrinkage_f1_beta22``) maximises ``F1(t) * Beta(t; 2, 2)`` (posterior
mode under a true Beta(2,2) prior with F1 likelihood, mode shrunk toward 0.5);
its results are reported as a robustness check, and the primary hypotheses are
evaluated on the L1-shrinkage variant (RQ61's explicit prediction).

For each ``lambda``: draw B=1000 bootstrap resamples (seed=42, n=77 with
replacement) ONCE and reuse the SAME resample indices for all lambdas (paired
comparison, RQ48/RQ61 design). On each resample: calibrate the combined
threshold on the in-bag windows, evaluate the corrected-router cpWER on the
out-of-bag (OOB) windows (RQ44's ``out_of_bag_cpwer``). Aggregate: threshold
median, 2.5/97.5 percentile interval width, mode count (RQ48's ``count_modes``,
>= 5% frequency -- the established RQ44/RQ48/RQ54/RQ61 mode definition), the
OOB cpWER median + 2.5/97.5 percentile width, and Hartigan's dip statistic
(the issue's multimodality diagnostic; ``dip > 0.05 -> multimodal``).

Pre-registered hypotheses (issue #994)
--------------------------------------
- H66a: Shrinkage+F1 threshold distribution has <= 2 modes. KILL if > 2 modes
        (>= 5% frequency; RQ48's ``count_modes`` definition, consistent with the
        RQ44->RQ48->RQ61 lineage). Hartigan's dip test is reported as a secondary
        multimodality diagnostic.
- H66b: Shrinkage+F1 OOB cpWER < 1.056 (RQ44 baseline). KILL if >= 1.056.
- H66c: Shrinkage+F1 2.5/97.5 width < 0.2489 (RQ54 F1 razor-thin cpWER CI width).
        KILL if >= 0.2489. The "2.5/97.5 width" is the OOB cpWER percentile
        width (the like-for-like comparison with RQ54's cpWER CI width); the
        threshold percentile width is also reported for traceability.

This script is pure reanalysis (numpy + stdlib; scipy is OPTIONAL, used only for
Hartigan's dip test with a try/except guard). Detector primitives, the bootstrap
draw, the OOB cpWER evaluator, and the baseline calibration rule are imported
verbatim from RQ44; F1 calibration and ``count_modes`` are imported verbatim from
RQ48.

Label: experimental/frontier. Builds on RQ13 (PR #904), RQ16 (PR #912), RQ25
(PR #929), RQ44 (PR #963), RQ48 (PR #965), RQ54 (PR #971), RQ61 (PR #991).
Closes #994.
"""
from __future__ import annotations

import csv
import json
import math
import sys
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
OUT_DIR = PROJECT_ROOT / "results" / "frontier" / "shrinkage_f1_combined_calibration"
OUT_CSV = OUT_DIR / "shrinkage_f1_combined_results.csv"
OUT_JSON = OUT_DIR / "shrinkage_f1_combined_results.json"

# ------------------------------------------ import RQ44's framework (verbatim reuse)
_RQ44_DIR = PROJECT_ROOT / "results" / "frontier" / "bootstrap_threshold_stability"
sys.path.insert(0, str(_RQ44_DIR))
import bootstrap_threshold_analysis as rq44  # noqa: E402  (path-injected import)

# ------------------------------------------ import RQ48's F1 + count_modes (verbatim reuse)
_RQ48_DIR = PROJECT_ROOT / "results" / "frontier" / "calibration_rule_comparison"
sys.path.insert(0, str(_RQ48_DIR))
import calibration_rule_analysis as rq48  # noqa: E402  (path-injected import)

# ------------------------------------------------------------------ constants
# Re-export RQ44's constants so thresholds are on the identical grid and the
# hallucination label matches RQ44/RQ48/RQ61 exactly (37 hallucinated / 40 clean).
THRESHOLD_GRID = rq44.THRESHOLD_GRID            # 0.00, 0.01, ..., 2.00 (201 pts)
EPS = rq44.EPS                                  # 1e-9
CATASTROPHIC_CPWER = rq44.CATASTROPHIC_CPWER    # 1.0
TARGET_SPECIFICITY = rq44.TARGET_SPECIFICITY    # 0.90 (kept for reference; F1 has no spec gate)

N_BOOT = 1000           # B=1000 (issue #994 specifies B=1000; RQ44 used 10000, RQ48 2000, RQ61 10000)
SEED = 42
MIN_MODE_FRACTION = 0.05   # "mode" = distinct threshold with >= 5% frequency (RQ48)

# Shrinkage prior: RQ44's bootstrap median threshold (0.380), reused by RQ61.
PRIOR_MEAN = 0.38
LAMBDAS = [0.0, 0.01, 0.1, 0.5, 1.0]

# Hartigan's dip-test multimodality flag (issue #994: "dip > 0.05 -> multimodal").
DIP_MULTIMODAL_THRESHOLD = 0.05

# RQ44 / RQ54 reference values -- the baselines to beat.
RQ44_OOB_CPWER_MEDIAN = 1.056      # RQ44's median held-out cpWER (H66b baseline)
RQ54_F1_CPWER_CI_WIDTH = 0.2489    # RQ54's F1-calibrated cascade BCa cpWER CI width (H66c baseline)
RQ44_N_MODES_5PCT = 5              # RQ44's mode count (>= 5%) under the baseline rule
RQ44_INTERVAL_WIDTH = 0.94         # RQ44's threshold 2.5/97.5 percentile interval width
RQ48_F1_MODES = 2                  # RQ48's F1 mode count on lang-id-entropy (B=2000)
RQ61_SHRINKAGE_MODES = 3           # RQ61's shrinkage mode count at lambda=1.0 (B=10000)

# Hypothesis kill thresholds (issue #994).
H66A_MAX_MODES = 2      # H66a: kill if > 2 modes (>= 5% frequency)
H66B_MAX_CPWER = 1.056  # H66b: kill if OOB cpWER >= 1.056 (supported if < 1.056, strict)
H66C_MAX_WIDTH = 0.2489 # H66c: kill if OOB cpWER 2.5/97.5 width >= 0.2489 (strict)
OOB_CPWER_GOOD = 1.10   # "good" OOB cpWER threshold (RQ44's H44c kill line)

# Detector + bootstrap framework reused verbatim from RQ44.
max_across_speakers = rq44.max_across_speakers
bootstrap_indices = rq44.bootstrap_indices
out_of_bag_cpwer = rq44.out_of_bag_cpwer
percentile_interval = rq44.percentile_interval

# F1 calibration + mode counter reused verbatim from RQ48.
calibrate_f1 = rq48.calibrate_f1
count_modes = rq48.count_modes


# --------------------------------------------------------------- confusion helper
def _confusion_arrays(
    scores: np.ndarray, labels: np.ndarray, grid_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Vectorised confusion-matrix sweep over ``grid_arr`` (faithful copy of
    RQ48/RQ61's helper).

    For each grid threshold ``t`` (ascending), flag = ``score >= t`` (with EPS
    tolerance, matching RQ44). Returns ``(tp, fp, tn, fn, n_pos, n_neg)`` as
    int arrays of shape ``(len(grid),)``."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    grid_arr = np.asarray(grid_arr, dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = int(len(pos))
    n_neg = int(len(neg))
    if n_pos > 0:
        tp = (pos[None, :] >= grid_arr[:, None] - EPS).sum(axis=1).astype(int)
    else:
        tp = np.zeros(grid_arr.shape, dtype=int)
    if n_neg > 0:
        fp = (neg[None, :] >= grid_arr[:, None] - EPS).sum(axis=1).astype(int)
    else:
        fp = np.zeros(grid_arr.shape, dtype=int)
    tn = n_neg - fp
    fn = n_pos - tp
    return tp, fp, tn, fn, n_pos, n_neg


def _sens_spec(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray,
    n_pos: int, n_neg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity / specificity arrays with safe division (faithful copy of
    RQ48/RQ61's helper)."""
    safe_pos = n_pos if n_pos > 0 else 1
    safe_neg = n_neg if n_neg > 0 else 1
    sens = tp / safe_pos if n_pos > 0 else np.zeros(tp.shape, dtype=float)
    spec = tn / safe_neg if n_neg > 0 else np.ones(tn.shape, dtype=float)
    return sens, spec


def _f1_array(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, sens: np.ndarray
) -> np.ndarray:
    """F1 = 2*prec*rec/(prec+rec) over the grid (RQ48's exact computation).

    precision = tp/(tp+fp) (0 when tp+fp == 0); recall = sensitivity. F1 is 0
    when prec+rec == 0. Faithful copy of RQ48's ``calibrate_f1`` arithmetic."""
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    denom_prec = tp + fp
    safe_prec_denom = np.where(denom_prec > 0, denom_prec, 1)
    prec = np.where(denom_prec > 0, tp / safe_prec_denom, 0.0)
    rec = np.asarray(sens, dtype=float)
    denom_f1 = prec + rec
    safe_f1_denom = np.where(denom_f1 > 0, denom_f1, 1.0)
    return np.where(denom_f1 > 0, 2.0 * prec * rec / safe_f1_denom, 0.0)


# --------------------------------------------------------------- combined objective
def shrinkage_f1_objective(
    threshold: float, f1: float, prior_mean: float, lam: float
) -> float:
    """The combined regularised objective: ``F1 - lam * |t - prior|``.

    The penalty pulls the threshold toward ``prior_mean``: when two thresholds
    tie (or nearly tie) on F1, the one closer to the prior has the higher
    objective. At ``lam = 0`` the objective reduces to pure F1 (RQ48's rule)."""
    return float(f1) - float(lam) * abs(float(threshold) - float(prior_mean))


# --------------------------------------------------------------- combined selection
def _select_shrinkage_f1(
    grid_arr: np.ndarray,
    f1: np.ndarray,
    sens: np.ndarray,
    spec: np.ndarray,
    tp: np.ndarray,
    fp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    prior_mean: float,
    lam: float,
) -> dict[str, Any]:
    """Pick the grid threshold maximising ``F1 - lam * |t - prior|``.

    Tie-break: higher objective -> higher F1 -> lower threshold (the last via
    the first True in the ascending grid, matching RQ48's lowest-threshold
    convention). At ``lam = 0`` this matches RQ48's ``calibrate_f1`` exactly
    (max F1, tie-break lowest threshold), so the combined rule is a strict
    generalisation of RQ48's F1 rule."""
    penalty = lam * np.abs(grid_arr - prior_mean)
    objective = f1 - penalty
    best_val = float(np.max(objective))
    # Tie-break 1: highest objective (within EPS).
    tie1 = objective >= best_val - EPS
    # Tie-break 2: highest F1.
    f1_masked = np.where(tie1, f1, -np.inf)
    best_f1 = float(np.max(f1_masked))
    tie2 = tie1 & (f1 >= best_f1 - EPS)
    # Tie-break 3: lowest threshold (first True in ascending grid).
    idx = int(np.argmax(tie2))
    return {
        "threshold": float(grid_arr[idx]),
        "sensitivity": float(sens[idx]),
        "specificity": float(spec[idx]),
        "precision": float(tp[idx] / (tp[idx] + fp[idx])) if (tp[idx] + fp[idx]) > 0 else 0.0,
        "f1": float(f1[idx]),
        "tp": int(tp[idx]),
        "fp": int(fp[idx]),
        "tn": int(tn[idx]),
        "fn": int(fn[idx]),
        "objective": float(objective[idx]),
        "penalty": float(penalty[idx]),
        "lambda": float(lam),
        "prior_mean": float(prior_mean),
    }


def calibrate_shrinkage_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    prior_mean: float = PRIOR_MEAN,
    lam: float = 0.0,
    grid: list[float] | None = None,
) -> dict[str, Any]:
    """Combined shrinkage+F1 calibration: maximise ``F1 - lam*|t - prior_mean|``
    over the threshold grid.

    This is the literal combination RQ61's FINDINGS predicted: "maximise
    ``F1 - lam*|t - 0.38|``". At ``lam = 0`` it reproduces RQ48's
    ``calibrate_f1`` exactly (max F1, lowest-threshold tie-break) -- the
    comparison's anchor. At ``lam > 0`` the L1 penalty (log-Laplace prior)
    pulls the threshold toward ``prior_mean`` (0.38), breaking F1 ties (and
    near-ties) in favour of thresholds closer to the prior."""
    if grid is None:
        grid = THRESHOLD_GRID
    grid_arr = np.asarray(grid, dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    tp, fp, tn, fn, n_pos, n_neg = _confusion_arrays(scores, labels, grid_arr)
    sens, spec = _sens_spec(tp, fp, tn, fn, n_pos, n_neg)
    f1 = _f1_array(tp, fp, fn, sens)
    return _select_shrinkage_f1(
        grid_arr, f1, sens, spec, tp, fp, tn, fn, prior_mean, lam
    )


# --------------------------------------------------------------- secondary: Beta(2,2) variant
def calibrate_shrinkage_f1_beta22(
    scores: np.ndarray,
    labels: np.ndarray,
    grid: list[float] | None = None,
) -> dict[str, Any]:
    """Secondary (issue-literal) variant: posterior mode under a true Beta(2,2)
    prior with F1 as the (unnormalised) likelihood.

    posterior(t) proportional to F1(t) * Beta(t; 2, 2) = F1(t) * 6 * t * (1 - t)
    for t in [0, 1]; the prior density is 0 outside [0, 1] so the posterior mode
    lies in [0, 1]. The Beta(2, 2) prior (mode 0.5, concentration 4) shrinks the
    threshold toward 0.5, penalising thresholds near 0 and 1 -- a symmetric
    Bayesian shrinkage. This honours the issue's literal "Beta(2, 2) prior,
    posterior mode" phrasing; the primary hypotheses use the L1-shrinkage variant
    (``calibrate_shrinkage_f1``), which is RQ61's explicit prediction.

    Tie-break: higher posterior -> higher F1 -> lower threshold (RQ48 convention)."""
    if grid is None:
        grid = THRESHOLD_GRID
    grid_arr = np.asarray(grid, dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    tp, fp, tn, fn, n_pos, n_neg = _confusion_arrays(scores, labels, grid_arr)
    sens, spec = _sens_spec(tp, fp, tn, fn, n_pos, n_neg)
    f1 = _f1_array(tp, fp, fn, sens)
    # Beta(2,2) density ∝ t*(1-t) on [0,1]; 0 outside. Posterior ∝ F1 * prior.
    prior = np.where((grid_arr >= 0.0) & (grid_arr <= 1.0),
                     grid_arr * (1.0 - grid_arr), 0.0)
    posterior = f1 * prior
    best_val = float(np.max(posterior))
    tie1 = posterior >= best_val - EPS
    f1_masked = np.where(tie1, f1, -np.inf)
    best_f1 = float(np.max(f1_masked))
    tie2 = tie1 & (f1 >= best_f1 - EPS)
    idx = int(np.argmax(tie2))
    return {
        "threshold": float(grid_arr[idx]),
        "sensitivity": float(sens[idx]),
        "specificity": float(spec[idx]),
        "precision": float(tp[idx] / (tp[idx] + fp[idx])) if (tp[idx] + fp[idx]) > 0 else 0.0,
        "f1": float(f1[idx]),
        "tp": int(tp[idx]),
        "fp": int(fp[idx]),
        "tn": int(tn[idx]),
        "fn": int(fn[idx]),
        "posterior": float(posterior[idx]),
        "prior_density": float(prior[idx]),
        "prior": "Beta(2,2)",
        "prior_mode": 0.5,
    }


# --------------------------------------------------------------- Hartigan's dip test
def hartigans_dip(values: np.ndarray) -> dict[str, Any]:
    """Hartigan's dip statistic (multimodality diagnostic, issue #994).

    Returns ``{dip, multimodal, pvalue, method}`` where ``multimodal = dip >
    0.05`` (the issue's rule). Uses ``scipy.stats.dip_test`` when available
    (scipy >= 1.10); falls back to ``method = "scipy_unavailable"`` with
    ``dip = None`` otherwise (the mode count from ``count_modes`` remains the
    primary H66a kill condition in that case). ``values`` with < 3 points has
    dip 0 (trivially unimodal)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return {"dip": 0.0, "multimodal": False, "pvalue": 1.0,
                "method": "trivial_lt3"}
    try:
        from scipy.stats import dip_test as _scipy_dip_test  # type: ignore
    except ImportError:
        return {"dip": None, "multimodal": None, "pvalue": None,
                "method": "scipy_unavailable"}
    res = _scipy_dip_test(arr)
    dip = float(res.statistic)
    pval = float(res.pvalue)
    return {"dip": dip,
            "multimodal": bool(dip > DIP_MULTIMODAL_THRESHOLD),
            "pvalue": pval,
            "method": "scipy_stats_dip_test"}


# --------------------------------------------------------------- per-lambda aggregation
def _summarise_lambda(
    thresholds: np.ndarray, oob_cpwer: np.ndarray
) -> dict[str, Any]:
    """Aggregate the bootstrap threshold + OOB cpWER distributions for one
    lambda. Mirrors RQ48/RQ61's ``_summarise_*`` so the output schema is directly
    comparable to RQ44/RQ48/RQ61."""
    thr = np.asarray(thresholds, dtype=float)
    oob_all = np.asarray(oob_cpwer, dtype=float)
    valid = ~np.isnan(oob_all)
    oob = oob_all[valid]
    thr_lo, thr_hi = percentile_interval(thr, 2.5, 97.5)
    oob_lo, oob_hi = percentile_interval(oob, 2.5, 97.5)
    modes = count_modes(thr, MIN_MODE_FRACTION)
    dip = hartigans_dip(thr)
    rq44_dist = rq44.threshold_distribution(thr)
    return {
        "threshold_distribution": {
            "median": round(float(np.median(thr)), 6),
            "mean": round(float(np.mean(thr)), 6),
            "std": round(float(np.std(thr)), 6),
            "min": round(float(np.min(thr)), 6),
            "max": round(float(np.max(thr)), 6),
            "percentile_2_5": round(float(thr_lo), 6),
            "percentile_97_5": round(float(thr_hi), 6),
            "interval_width": round(float(thr_hi - thr_lo), 6),
            "n_unique": int(np.unique(thr).size),
            "n_modes_5pct": modes["n_modes"],
            "modes_5pct": modes["modes"],
            "min_mode_fraction": float(MIN_MODE_FRACTION),
            "modes_within_10pct_max": [
                {"threshold": round(t, 6), "count": c, "fraction": round(f, 6)}
                for t, c, f in rq44_dist["modes_within_10pct"]
            ],
            "top_values": [
                {"threshold": round(t, 6), "count": c, "fraction": round(f, 6)}
                for t, c, f in rq44_dist["top_values"]
            ],
            "hartigans_dip": dip,
        },
        "oob_cpwer_distribution": {
            "n_valid": int(valid.sum()),
            "median": round(float(np.median(oob)), 6),
            "mean": round(float(np.mean(oob)), 6),
            "min": round(float(np.min(oob)), 6),
            "max": round(float(np.max(oob)), 6),
            "percentile_2_5": round(float(oob_lo), 6),
            "percentile_97_5": round(float(oob_hi), 6),
            "interval_width": round(float(oob_hi - oob_lo), 6),
            "frac_below_1_10": round(float(np.mean(oob < OOB_CPWER_GOOD)), 6),
            "frac_below_rq44_median": round(
                float(np.mean(oob < RQ44_OOB_CPWER_MEDIAN)), 6),
        },
    }


# --------------------------------------------------------------- best-lambda selection
def select_best_lambda(per_lambda_summary: dict[str, Any]) -> dict[str, Any]:
    """Pick the lambda that best reduces modality while maintaining
    deployability.

    Criteria (in order, mirroring RQ61):
    1. Deployable: OOB median cpWER < 1.056 (RQ44's median; H66b boundary).
    2. Among deployable: fewest modes (n_modes_5pct).
    3. Tie-break: narrowest THRESHOLD interval width (RQ61's stability criterion).
    4. Tie-break: smallest lambda (least regularisation, most faithful to data).

    If NO lambda is deployable, pick the one whose OOB cpWER is closest to
    (and <) 1.056, then fewest modes, then narrowest width, then smallest lambda.
    ``per_lambda_summary`` is a dict keyed by lambda-string with values carrying
    ``lambda``, ``n_modes_5pct``, ``threshold_interval_width``,
    ``oob_cpwer_median``, ``oob_cpwer_interval_width``."""
    items = list(per_lambda_summary.items())
    deployable = [
        (k, v) for k, v in items
        if float(v["oob_cpwer_median"]) < H66B_MAX_CPWER - EPS
    ]
    if deployable:
        best_key, best_v = min(
            deployable,
            key=lambda kv: (
                int(kv[1]["n_modes_5pct"]),
                float(kv[1]["threshold_interval_width"]),
                float(kv[1]["lambda"]),
            ),
        )
        reason = ("deployable (OOB < 1.056); selected by fewest modes, then "
                  "narrowest threshold width, then smallest lambda")
    else:
        best_key, best_v = min(
            items,
            key=lambda kv: (
                float(kv[1]["oob_cpwer_median"]),
                int(kv[1]["n_modes_5pct"]),
                float(kv[1]["threshold_interval_width"]),
                float(kv[1]["lambda"]),
            ),
        )
        reason = ("no lambda deployable (all OOB >= 1.056); selected by closest "
                  "OOB cpWER to 1.056, then fewest modes, then narrowest width, "
                  "then smallest lambda")
    all_supported = (
        _h66a_supported(best_v["n_modes_5pct"])
        and _h66b_supported(best_v["oob_cpwer_median"])
        and _h66c_supported(best_v["oob_cpwer_interval_width"])
    )
    return {
        "lambda": float(best_v["lambda"]),
        "lambda_key": best_key,
        "reason": reason,
        "all_hypotheses_supported": bool(all_supported),
        "n_modes_5pct": int(best_v["n_modes_5pct"]),
        "threshold_interval_width": float(best_v["threshold_interval_width"]),
        "oob_cpwer_median": float(best_v["oob_cpwer_median"]),
        "oob_cpwer_interval_width": float(best_v["oob_cpwer_interval_width"]),
    }


# --------------------------------------------------------------- hypothesis helpers
def _h66a_supported(n_modes: int) -> bool:
    """H66a: <= 2 modes supported; > 2 killed."""
    return int(n_modes) <= H66A_MAX_MODES


def _h66b_supported(oob_median: float) -> bool:
    """H66b: OOB cpWER < 1.056 supported; >= 1.056 killed (strict)."""
    return float(oob_median) < H66B_MAX_CPWER - EPS


def _h66c_supported(oob_width: float) -> bool:
    """H66c: OOB cpWER 2.5/97.5 width < 0.2489 supported; >= 0.2489 killed (strict)."""
    return float(oob_width) < H66C_MAX_WIDTH - EPS


# --------------------------------------------------------------------- driver
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    windows = data["windows"]
    n = len(windows)

    # Per-window signals (identical to RQ44/RQ48/RQ61).
    lang_ent = np.array([max_across_speakers(w) for w in windows], dtype=float)
    mixed_cpwer = np.array([float(w["always_mixed_cpwer"]) for w in windows], dtype=float)
    sep_cpwer = np.array([float(w["always_separated_cpwer"]) for w in windows], dtype=float)
    labels = (sep_cpwer > CATASTROPHIC_CPWER).astype(int)  # 1 = hallucinated
    n_hall = int(labels.sum())
    n_clean = int((labels == 0).sum())
    grid_arr = np.asarray(THRESHOLD_GRID, dtype=float)

    # ----------------------------------------------- in-sample calibration per lambda
    in_sample: dict[str, Any] = {}
    for lam in LAMBDAS:
        cal = calibrate_shrinkage_f1(lang_ent, labels, prior_mean=PRIOR_MEAN, lam=lam)
        flag = lang_ent >= cal["threshold"] - EPS
        selected = np.where(flag, mixed_cpwer, sep_cpwer)
        in_sample[str(lam)] = {
            "lambda": lam,
            "threshold": round(float(cal["threshold"]), 6),
            "sensitivity": round(float(cal["sensitivity"]), 6),
            "specificity": round(float(cal["specificity"]), 6),
            "precision": round(float(cal["precision"]), 6),
            "f1": round(float(cal["f1"]), 6),
            "tp": int(cal["tp"]), "fp": int(cal["fp"]),
            "tn": int(cal["tn"]), "fn": int(cal["fn"]),
            "penalty": round(float(cal["penalty"]), 6),
            "objective": round(float(cal["objective"]), 6),
            "expected_cpwer": round(float(selected.mean()), 6),
        }

    # In-sample secondary (Beta(2,2)) variant.
    cal_beta = calibrate_shrinkage_f1_beta22(lang_ent, labels)
    flag_beta = lang_ent >= cal_beta["threshold"] - EPS
    selected_beta = np.where(flag_beta, mixed_cpwer, sep_cpwer)
    in_sample_beta22 = {
        "threshold": round(float(cal_beta["threshold"]), 6),
        "sensitivity": round(float(cal_beta["sensitivity"]), 6),
        "specificity": round(float(cal_beta["specificity"]), 6),
        "precision": round(float(cal_beta["precision"]), 6),
        "f1": round(float(cal_beta["f1"]), 6),
        "tp": int(cal_beta["tp"]), "fp": int(cal_beta["fp"]),
        "tn": int(cal_beta["tn"]), "fn": int(cal_beta["fn"]),
        "expected_cpwer": round(float(selected_beta.mean()), 6),
        "prior": "Beta(2,2)",
        "prior_mode": 0.5,
    }

    # ----------------------------------------------------------------- bootstrap
    # ONE index draw, reused for all lambdas -> paired comparison (RQ48/RQ61 design).
    boot_idx = bootstrap_indices(n, N_BOOT, SEED)  # (N_BOOT, n)
    per_lambda: dict[str, Any] = {}
    for lam in LAMBDAS:
        per_lambda[str(lam)] = {
            "thresholds": np.empty(N_BOOT, dtype=float),
            "oob_cpwer": np.empty(N_BOOT, dtype=float),
            "n_oob": np.empty(N_BOOT, dtype=int),
        }

    for b in range(N_BOOT):
        idx = boot_idx[b]
        # Confusion arrays depend only on the resample, not on lambda -> compute
        # once per resample and reuse for all lambdas.
        tp, fp, tn, fn, n_pos, n_neg = _confusion_arrays(
            lang_ent[idx], labels[idx], grid_arr
        )
        sens, spec = _sens_spec(tp, fp, tn, fn, n_pos, n_neg)
        f1 = _f1_array(tp, fp, fn, sens)
        for lam in LAMBDAS:
            cal = _select_shrinkage_f1(
                grid_arr, f1, sens, spec, tp, fp, tn, fn,
                PRIOR_MEAN, lam,
            )
            thr = float(cal["threshold"])
            oob = out_of_bag_cpwer(lang_ent, mixed_cpwer, sep_cpwer, thr, idx)
            slot = per_lambda[str(lam)]
            slot["thresholds"][b] = thr
            slot["oob_cpwer"][b] = oob["cpwer"]
            slot["n_oob"][b] = oob["n_oob"]

    # ----------------------------------------------- aggregate per lambda
    per_lambda_summary: dict[str, Any] = {}
    per_lambda_full: dict[str, Any] = {}
    for lam in LAMBDAS:
        key = str(lam)
        thr = per_lambda[key]["thresholds"]
        oob = per_lambda[key]["oob_cpwer"]
        n_oob_mean = float(np.mean(per_lambda[key]["n_oob"]))
        summary = _summarise_lambda(thr, oob)
        td = summary["threshold_distribution"]
        od = summary["oob_cpwer_distribution"]
        h66a = _h66a_supported(td["n_modes_5pct"])
        h66b = _h66b_supported(od["median"])
        h66c = _h66c_supported(od["interval_width"])
        per_lambda_summary[key] = {
            "lambda": lam,
            "n_modes_5pct": td["n_modes_5pct"],
            "threshold_interval_width": td["interval_width"],
            "oob_cpwer_median": od["median"],
            "oob_cpwer_interval_width": od["interval_width"],
            "h66a_supported": bool(h66a),
            "h66b_supported": bool(h66b),
            "h66c_supported": bool(h66c),
        }
        per_lambda_full[key] = {
            "lambda": lam,
            "in_sample": in_sample[key],
            "summary": summary,
            "mean_oob_size": round(n_oob_mean, 4),
            "hypothesis_verdicts": {
                "H66a": {
                    "statement": f"<= {H66A_MAX_MODES} modes (vs RQ44's {RQ44_N_MODES_5PCT})",
                    "n_modes_5pct": td["n_modes_5pct"],
                    "max_modes": H66A_MAX_MODES,
                    "kill": f"> {H66A_MAX_MODES} modes with >= 5% frequency",
                    "supported": bool(h66a),
                    "hartigans_dip": td["hartigans_dip"],
                },
                "H66b": {
                    "statement": f"OOB cpWER < {H66B_MAX_CPWER} (RQ44 baseline)",
                    "median_oob_cpwer": od["median"],
                    "rq44_median_oob_cpwer": RQ44_OOB_CPWER_MEDIAN,
                    "kill": f"OOB cpWER >= {H66B_MAX_CPWER}",
                    "supported": bool(h66b),
                },
                "H66c": {
                    "statement": (f"OOB cpWER 2.5/97.5 width < {H66C_MAX_WIDTH} "
                                  f"(RQ54 F1 cpWER CI width)"),
                    "oob_cpwer_interval_width": od["interval_width"],
                    "rq54_f1_cpwer_ci_width": RQ54_F1_CPWER_CI_WIDTH,
                    "threshold_interval_width": td["interval_width"],
                    "kill": f"OOB cpWER width >= {H66C_MAX_WIDTH}",
                    "supported": bool(h66c),
                },
            },
            "per_bootstrap": {
                "thresholds": [round(float(t), 6) for t in thr],
                "oob_cpwer": [round(float(c), 6) if not math.isnan(float(c)) else None
                              for c in oob],
                "n_oob": [int(x) for x in per_lambda[key]["n_oob"]],
            },
        }

    # ------------------------------------------------------------ best lambda
    best = select_best_lambda(per_lambda_summary)
    best_key = best["lambda_key"]
    best_h66a = per_lambda_summary[best_key]["h66a_supported"]
    best_h66b = per_lambda_summary[best_key]["h66b_supported"]
    best_h66c = per_lambda_summary[best_key]["h66c_supported"]
    best_dip = per_lambda_full[best_key]["summary"]["threshold_distribution"]["hartigans_dip"]

    # ----------------------------------------------- secondary Beta(2,2) bootstrap
    boot_thr_beta = np.empty(N_BOOT, dtype=float)
    boot_oob_beta = np.empty(N_BOOT, dtype=float)
    boot_n_oob_beta = np.empty(N_BOOT, dtype=int)
    for b in range(N_BOOT):
        idx = boot_idx[b]
        cal_b = calibrate_shrinkage_f1_beta22(lang_ent[idx], labels[idx])
        thr = float(cal_b["threshold"])
        oob = out_of_bag_cpwer(lang_ent, mixed_cpwer, sep_cpwer, thr, idx)
        boot_thr_beta[b] = thr
        boot_oob_beta[b] = oob["cpwer"]
        boot_n_oob_beta[b] = oob["n_oob"]
    beta_summary = _summarise_lambda(boot_thr_beta, boot_oob_beta)
    beta_td = beta_summary["threshold_distribution"]
    beta_od = beta_summary["oob_cpwer_distribution"]
    beta_h66a = _h66a_supported(beta_td["n_modes_5pct"])
    beta_h66b = _h66b_supported(beta_od["median"])
    beta_h66c = _h66c_supported(beta_od["interval_width"])

    summary: dict[str, Any] = {
        "label": "experimental/frontier",
        "rq": ("RQ66: Shrinkage + F1 combined threshold calibration "
               "(Bayesian shrinkage prior on the lang-id entropy threshold "
               "combined with the F1 calibration rule)"),
        "closes_issue": 994,
        "builds_on": {
            "RQ13": "results/frontier/diverse_hallucination_detector/ (PR #904)",
            "RQ16": "results/frontier/corrected_router_simulation/ (PR #912)",
            "RQ25": "results/frontier/out_of_sample_router/ (PR #929)",
            "RQ44": "results/frontier/bootstrap_threshold_stability/ (PR #963)",
            "RQ48": "results/frontier/calibration_rule_comparison/ (PR #965)",
            "RQ54": "results/frontier/cascade_f1_calibration/ (PR #971)",
            "RQ61": "results/frontier/shrinkage_threshold_calibration/ (PR #991)",
        },
        "source_data": str(SRC_JSON.relative_to(PROJECT_ROOT)),
        "source_label": "external/sanity-check",
        "method": (
            "reanalysis only (no Whisper / no ASR / no LLM); B=1000 bootstrap "
            "resamples (seed=42) of the 77 AISHELL-4 windows, drawn ONCE and "
            "reused for all 5 lambda values + the Beta(2,2) variant (paired "
            "comparison). On each resample: calibrate the combined shrinkage+F1 "
            "threshold (maximise F1(t) - lambda*|t - 0.38| over the grid) on "
            "in-bag windows, evaluate corrected-router cpWER on the out-of-bag "
            "windows (RQ44's out_of_bag_cpwer). At lambda=0 the rule reproduces "
            "RQ48's calibrate_f1 exactly. The L1 penalty is the log-Laplace "
            "prior, so the combined estimate is the MAP (posterior mode) under a "
            "Laplace shrinkage prior centred at 0.38 with F1 as the likelihood -- "
            "the literal realisation of RQ61's predicted 'F1 - lambda*|t - 0.38|' "
            "combination. A secondary Beta(2,2) variant (posterior mode under a "
            "true Beta(2,2) prior) honours the issue's literal phrasing and is "
            "reported as a robustness check. Detector, bootstrap draw, OOB "
            "evaluator imported verbatim from RQ44; F1 + count_modes from RQ48."
        ),
        "meeting_id": data["meeting_id"],
        "n_windows": n,
        "n_hallucinated": n_hall,
        "n_clean": n_clean,
        "hallucination_label_rule": "always_separated_cpwer > 1.0",
        "routing_rule": (
            "lang_id_entropy >= threshold -> route MIXED (always_mixed_cpwer); "
            "else route SEPARATED (always_separated_cpwer). HIGH lang-id entropy "
            "= diverse multilingual gibberish = hallucination (RQ13/RQ16/RQ25/"
            "RQ44 convention)."
        ),
        "combined_rule": (
            "maximise F1(t) - lambda*|t - prior_mean| over the threshold grid, "
            "where F1 is RQ48's exact F1 score and prior_mean = 0.38 (RQ44's "
            "bootstrap median, RQ61's prior). Tie-break: higher objective, then "
            "higher F1, then lower threshold. lambda=0 reproduces RQ48's "
            "calibrate_f1 exactly. The L1 penalty is the log-Laplace prior -> "
            "the estimate is the MAP under a Laplace shrinkage prior."
        ),
        "secondary_beta22_rule": (
            "posterior mode of F1(t) * Beta(t; 2, 2) over t in [0, 1] (true "
            "Beta(2,2) prior with F1 likelihood, mode shrunk toward 0.5). "
            "Honours the issue's literal 'Beta(2,2) prior, posterior mode' "
            "phrasing; reported as a robustness check, not the primary method."
        ),
        "mode_definition": (
            "mode = distinct threshold value with bootstrap frequency >= 5% "
            "(RQ48 count_modes, min_fraction=0.05). This is the explicit "
            "kill-condition definition for H66a, consistent with RQ44/RQ48/RQ54/"
            "RQ61. Hartigan's dip test (dip > 0.05 -> multimodal) is reported as "
            "a secondary multimodality diagnostic."
        ),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "resample_size": n,
            "paired": True,
            "expected_oob_size": round(n * ((1 - 1 / n) ** n), 4),
        },
        "prior": {
            "prior_mean": PRIOR_MEAN,
            "prior_mean_source": "RQ44 bootstrap median threshold (0.380)",
            "lambdas": LAMBDAS,
            "bayesian_framing": ("L1 penalty = log-Laplace prior centred at 0.38; "
                                 "combined estimate = MAP (posterior mode)"),
        },
        "rq44_reference": {
            "median_oob_cpwer": RQ44_OOB_CPWER_MEDIAN,
            "n_modes_5pct": RQ44_N_MODES_5PCT,
            "threshold_interval_width": RQ44_INTERVAL_WIDTH,
            "baseline_rule": "max sensitivity at >= 90% specificity",
        },
        "rq48_f1_reference": {
            "n_modes_5pct": RQ48_F1_MODES,
            "note": "RQ48's F1 on lang-id-entropy gave 2 modes {0.38, 0.01} (B=2000).",
        },
        "rq61_shrinkage_reference": {
            "n_modes_5pct": RQ61_SHRINKAGE_MODES,
            "best_lambda": 1.0,
            "note": "RQ61's shrinkage at lambda=1.0 gave 3 modes {0.38, 0.84, 0.87} (B=10000).",
        },
        "rq54_f1_reference": {
            "f1_cpwer_ci_width": RQ54_F1_CPWER_CI_WIDTH,
            "note": ("RQ54's F1-calibrated cascade BCa cpWER CI width 0.2489 "
                     "(H66c baseline). RQ66's H66c compares the corrected-router "
                     "OOB cpWER 2.5/97.5 percentile width against this value."),
        },
        "in_sample_calibration": in_sample,
        "in_sample_beta22": in_sample_beta22,
        "per_lambda": {k: {kk: vv for kk, vv in v.items() if kk != "per_bootstrap"}
                       for k, v in per_lambda_full.items()},
        "best_lambda": best,
        "hypothesis_verdicts": {
            "H66a": {
                "statement": (f"Shrinkage+F1 threshold (best lambda={best['lambda']}) "
                              f"has <= {H66A_MAX_MODES} modes (vs RQ44's {RQ44_N_MODES_5PCT}, "
                              f"RQ48 F1's {RQ48_F1_MODES}, RQ61 shrinkage's {RQ61_SHRINKAGE_MODES})"),
                "best_lambda": best["lambda"],
                "n_modes_5pct": best["n_modes_5pct"],
                "max_modes": H66A_MAX_MODES,
                "kill": f"> {H66A_MAX_MODES} modes with >= 5% frequency",
                "supported": bool(best_h66a),
                "hartigans_dip": best_dip,
            },
            "H66b": {
                "statement": (f"Shrinkage+F1 OOB cpWER (best lambda={best['lambda']}) "
                              f"< {H66B_MAX_CPWER} (RQ44 baseline)"),
                "best_lambda": best["lambda"],
                "median_oob_cpwer": best["oob_cpwer_median"],
                "rq44_median_oob_cpwer": RQ44_OOB_CPWER_MEDIAN,
                "kill": f"OOB cpWER >= {H66B_MAX_CPWER}",
                "supported": bool(best_h66b),
            },
            "H66c": {
                "statement": (f"Shrinkage+F1 OOB cpWER 2.5/97.5 width (best lambda="
                              f"{best['lambda']}) < {H66C_MAX_WIDTH} (RQ54 F1 cpWER CI width)"),
                "best_lambda": best["lambda"],
                "oob_cpwer_interval_width": best["oob_cpwer_interval_width"],
                "rq54_f1_cpwer_ci_width": RQ54_F1_CPWER_CI_WIDTH,
                "threshold_interval_width": best["threshold_interval_width"],
                "kill": f"OOB cpWER width >= {H66C_MAX_WIDTH}",
                "supported": bool(best_h66c),
            },
        },
        "secondary_beta22": {
            "in_sample": in_sample_beta22,
            "summary": beta_summary,
            "mean_oob_size": round(float(np.mean(boot_n_oob_beta)), 4),
            "hypothesis_verdicts": {
                "H66a": {
                    "statement": f"Beta(2,2)+F1 threshold has <= {H66A_MAX_MODES} modes",
                    "n_modes_5pct": beta_td["n_modes_5pct"],
                    "supported": bool(beta_h66a),
                    "hartigans_dip": beta_td["hartigans_dip"],
                },
                "H66b": {
                    "statement": f"Beta(2,2)+F1 OOB cpWER < {H66B_MAX_CPWER}",
                    "median_oob_cpwer": beta_od["median"],
                    "supported": bool(beta_h66b),
                },
                "H66c": {
                    "statement": (f"Beta(2,2)+F1 OOB cpWER width < {H66C_MAX_WIDTH}"),
                    "oob_cpwer_interval_width": beta_od["interval_width"],
                    "supported": bool(beta_h66c),
                },
            },
        },
    }

    # ----------------------------------------------------------- write JSON
    summary_with_arrays: dict[str, Any] = dict(summary)
    summary_with_arrays["per_bootstrap"] = {
        k: per_lambda_full[k]["per_bootstrap"] for k in per_lambda_full
    }
    summary_with_arrays["per_bootstrap_beta22"] = {
        "thresholds": [round(float(t), 6) for t in boot_thr_beta],
        "oob_cpwer": [round(float(c), 6) if not math.isnan(float(c)) else None
                      for c in boot_oob_beta],
        "n_oob": [int(x) for x in boot_n_oob_beta],
    }
    OUT_JSON.write_text(
        json.dumps(summary_with_arrays, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------- write CSV
    csv_fields = [
        "lambda", "in_sample_threshold", "in_sample_expected_cpwer", "in_sample_f1",
        "thr_median", "thr_p2_5", "thr_p97_5", "thr_interval_width",
        "thr_n_unique", "thr_n_modes_5pct",
        "oob_cpwer_median", "oob_cpwer_mean", "oob_cpwer_p2_5", "oob_cpwer_p97_5",
        "oob_cpwer_interval_width", "oob_frac_below_1_10", "oob_frac_below_rq44",
        "H66a_supported", "H66b_supported", "H66c_supported",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields)
        wr.writeheader()
        for lam in LAMBDAS:
            key = str(lam)
            pf = per_lambda_full[key]
            s = pf["summary"]
            td = s["threshold_distribution"]
            od = s["oob_cpwer_distribution"]
            wr.writerow({
                "lambda": lam,
                "in_sample_threshold": in_sample[key]["threshold"],
                "in_sample_expected_cpwer": in_sample[key]["expected_cpwer"],
                "in_sample_f1": in_sample[key]["f1"],
                "thr_median": td["median"],
                "thr_p2_5": td["percentile_2_5"],
                "thr_p97_5": td["percentile_97_5"],
                "thr_interval_width": td["interval_width"],
                "thr_n_unique": td["n_unique"],
                "thr_n_modes_5pct": td["n_modes_5pct"],
                "oob_cpwer_median": od["median"],
                "oob_cpwer_mean": od["mean"],
                "oob_cpwer_p2_5": od["percentile_2_5"],
                "oob_cpwer_p97_5": od["percentile_97_5"],
                "oob_cpwer_interval_width": od["interval_width"],
                "oob_frac_below_1_10": od["frac_below_1_10"],
                "oob_frac_below_rq44": od["frac_below_rq44_median"],
                "H66a_supported": "yes" if per_lambda_summary[key]["h66a_supported"] else "no",
                "H66b_supported": "yes" if per_lambda_summary[key]["h66b_supported"] else "no",
                "H66c_supported": "yes" if per_lambda_summary[key]["h66c_supported"] else "no",
            })

    # ----------------------------------------------------------- console
    print(f"=== RQ66: Shrinkage + F1 combined threshold calibration (AISHELL-4, {n} windows) ===")
    print(f"Label: experimental/frontier  |  Source: {SRC_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Hallucination label: always_separated_cpwer > 1.0 -> {n_hall} hall / {n_clean} clean")
    print(f"Bootstrap: B={N_BOOT}, seed={SEED}, paired across lambdas | prior_mean={PRIOR_MEAN}")
    print()
    print("In-sample calibration (full 77 windows, per lambda):")
    for lam in LAMBDAS:
        isam = in_sample[str(lam)]
        print(f"  lambda={lam:<4}  thr={isam['threshold']:.4f}  cpWER={isam['expected_cpwer']:.4f}"
              f"  F1={isam['f1']:.4f}  sens={isam['sensitivity']:.4f} spec={isam['specificity']:.4f}"
              f"  penalty={isam['penalty']:.4f}  obj={isam['objective']:.4f}")
    print(f"  Beta(2,2)   thr={in_sample_beta22['threshold']:.4f}  cpWER={in_sample_beta22['expected_cpwer']:.4f}"
          f"  F1={in_sample_beta22['f1']:.4f}  sens={in_sample_beta22['sensitivity']:.4f}")
    print()
    print(f"Bootstrap threshold + OOB cpWER distributions (B={N_BOOT}, paired):")
    for lam in LAMBDAS:
        key = str(lam)
        td = per_lambda_full[key]["summary"]["threshold_distribution"]
        od = per_lambda_full[key]["summary"]["oob_cpwer_distribution"]
        print(f"  --- lambda={lam} ---")
        print(f"    threshold: median={td['median']:.4f}  "
              f"pct[{td['percentile_2_5']:.4f}, {td['percentile_97_5']:.4f}]  "
              f"width={td['interval_width']:.4f}  "
              f"n_unique={td['n_unique']}  n_modes>=5%={td['n_modes_5pct']}")
        for m in td["modes_5pct"]:
            print(f"      mode thr={m['threshold']:.4f}  count={m['count']}  "
                  f"frac={m['fraction']:.3f}")
        dip = td["hartigans_dip"]
        dip_s = (f"dip={dip['dip']:.4f} multimodal={dip['multimodal']}"
                 if dip["dip"] is not None else f"dip={dip['method']}")
        print(f"    Hartigan dip: {dip_s}")
        print(f"    oob cpWER: median={od['median']:.4f}  mean={od['mean']:.4f}  "
              f"pct[{od['percentile_2_5']:.4f}, {od['percentile_97_5']:.4f}]  "
              f"width={od['interval_width']:.4f}  frac<1.10={od['frac_below_1_10']:.3f}")
    print()
    print(f"  --- Beta(2,2) variant ---")
    print(f"    threshold: median={beta_td['median']:.4f}  "
          f"pct[{beta_td['percentile_2_5']:.4f}, {beta_td['percentile_97_5']:.4f}]  "
          f"width={beta_td['interval_width']:.4f}  n_modes>=5%={beta_td['n_modes_5pct']}")
    for m in beta_td["modes_5pct"]:
        print(f"      mode thr={m['threshold']:.4f}  count={m['count']}  "
              f"frac={m['fraction']:.3f}")
    print(f"    oob cpWER: median={beta_od['median']:.4f}  "
          f"width={beta_od['interval_width']:.4f}")
    print()
    print(f"Best lambda: {best['lambda']}  (all hypotheses supported: "
          f"{best['all_hypotheses_supported']})")
    print(f"  reason: {best['reason']}")
    print()
    print("Hypothesis verdicts (at best lambda):")
    print(f"  H66a (<= {H66A_MAX_MODES} modes vs RQ44's {RQ44_N_MODES_5PCT}): "
          f"{'SUPPORTED' if best_h66a else 'KILLED'}  "
          f"(n_modes={best['n_modes_5pct']})")
    print(f"  H66b (OOB cpWER < {H66B_MAX_CPWER}):                "
          f"{'SUPPORTED' if best_h66b else 'KILLED'}  "
          f"(median OOB={best['oob_cpwer_median']:.4f})")
    print(f"  H66c (OOB cpWER width < {H66C_MAX_WIDTH}):          "
          f"{'SUPPORTED' if best_h66c else 'KILLED'}  "
          f"(width={best['oob_cpwer_interval_width']:.4f})")
    print()
    print(f"References: RQ44 {RQ44_N_MODES_5PCT} modes / OOB {RQ44_OOB_CPWER_MEDIAN} / thr width {RQ44_INTERVAL_WIDTH}; "
          f"RQ48 F1 {RQ48_F1_MODES} modes; RQ61 shrinkage {RQ61_SHRINKAGE_MODES} modes; "
          f"RQ54 F1 cpWER CI width {RQ54_F1_CPWER_CI_WIDTH}")
    print()
    print(f"Wrote: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
