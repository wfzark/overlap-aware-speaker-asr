"""RQ62: Cascade with KL+lang-id ensemble gate -- does a two-detector ensemble
produce a less aggressive or more robust cascade operating point than KL alone?

REANALYSIS ONLY -- no Whisper / no ASR / no LLM model is run. RQ59 (PR #980)
showed both Youden's J and F1 collapse to 83.1% escalation on the KL detector
because the KL ROC is flat-topped. RQ62 tests whether a KL+lang-id ENSEMBLE
gate (escalate if EITHER detector flags, OR-gate; or BOTH, AND-gate) produces a
different -- ideally less aggressive or more robust -- cascade operating point
than KL alone.

Label: experimental/frontier.

Controlled comparison design
----------------------------
The cascade simulation is held fixed at RQ43's actual implementation (RQ59 /
RQ54 convention) so the comparison to RQ43's 0.888947 anchor (H62b), RQ54's
83.1% F1 escalation (H62a), and RQ59's 83.1% Youden's J escalation is
apples-to-apples:

- Tier 1 (whisper-tiny) cpWER per window = RQ43's ``tiny_sep_cpwer`` (the real
  whisper-tiny separated-audio cpWER; == ``always_separated_cpwer`` in
  AISHELL-4, verified in tests).
- Tier 3 (whisper-base) cpWER per window = RQ43's ``base_sep_cpwer`` =
  ``tiny_sep_cpwer * 0.428031`` (the model_scale separated base/tiny CER
  ratio, constant across overlap). This is RQ43's actual base-cpWER estimate.
- Ensemble gate: escalate to base when (KL >= kl_thr) [OR|AND] (lang_id >=
  lang_thr). KL score = RQ58's 2-gram character KL divergence (threshold
  5.418144, calibrated at >=90% specificity on the 40 non-hallucinated tracks).
  lang-id score = RQ13's max-across-speakers script-category entropy
  (threshold 0.38, RQ44's >=90%-specificity calibration).

The hallucination label used to calibrate BOTH thresholds is
``tiny_sep_cpwer > 1.0`` (37 hallucinated / 40 clean), matching
RQ44/RQ48/RQ54/RQ59's label rule. Both detectors flag high-score windows as
hallucinated and escalate them to base -- the same direction as RQ43's cascade.

Ensemble gate logic
-------------------
- OR gate (primary): escalate if KL >= kl_thr OR lang_id >= lang_thr. This is
  the "either detector flags" gate from the issue. By construction it escalates
  AT LEAST as many windows as either detector alone.
- AND gate (secondary): escalate if KL >= kl_thr AND lang_id >= lang_thr. This
  is the conservative complement; it escalates AT MOST as many windows as
  either detector alone.

Method
------
1. Load RQ43's 77 per-window (tiny_sep_cpwer, base_sep_cpwer, kl_sep,
   window_id) and RQ58's 77 per-window (kl_score, lang_id_entropy,
   window_id); join by window_id (both are 0..76 in order, verified in tests).
2. Labels = tiny_sep_cpwer > 1.0 (37 hall / 40 clean).
3. In-sample calibration: KL threshold via RQ58's
   ``calibrate_threshold_at_specificity`` (candidate-set, >=90% specificity,
   smallest-threshold tie-break) -- reproduces 5.418144. lang-id threshold via
   RQ44's ``calibrate_threshold_at_spec`` (0.01-step grid [0.00, 2.00],
   >=90% specificity, max sensitivity, tie-break by higher spec then lower
   threshold) -- reproduces 0.38.
4. In-sample ensemble gate (OR and AND) at the calibrated thresholds ->
   cascade cpWER (theta_hat), escalation fraction, compute cost.
5. Bootstrap B=10000, seed=42: for each resample, re-calibrate BOTH thresholds
   on the in-bag windows (same calibration rules as in-sample) and evaluate the
   ensemble cascade cpWER (OR and AND) on the out-of-bag (OOB) windows (RQ44's
   OOB protocol). Records the per-resample (kl_thr, lang_thr) pair (for mode
   counting) and OOB cpWER (for the BCa CI).
6. Delete-1 jackknife (77 fits) for the BCa acceleration (separately for OR
   and AND).
7. BCa 95% CI on the OOB cpWER distribution (bias-corrected + accelerated;
   Acklam inverse-normal, no scipy) -- separately for OR and AND.
8. Mode count on the KL-threshold and lang-id-threshold bootstrap distributions
   (RQ48's ``count_modes``, min_fraction=0.05).
9. Pre-registered hypothesis verdicts H62a/b/c (primary on the OR gate; AND
   reported as the secondary gate).

Pre-registered hypotheses
-------------------------
- H62a: Ensemble cascade escalates < 83.1% of windows to base (less aggressive
        than KL-alone RQ59's 83.1%). Kill: in-sample escalation fraction >=
        0.831.
- H62b: Ensemble cascade OOB cpWER <= 0.889 (matches RQ43's original-rule
        cpWER 0.888947). Kill: OOB median cpWER > 0.889.
- H62c: Ensemble cascade BCa CI width <= 0.2489 (maintains robustness vs
        RQ46's original-rule width). Kill: BCa width > 0.2489.

The primary gate for H62a/b/c is the OR gate (the issue's "either detector
flags" ensemble). The AND gate is reported as a secondary analysis for
completeness; its verdicts are reported alongside but do not override the OR
verdicts.

Methodological note on the H62c comparison: RQ46's 0.2489 anchor was a
percentile CI evaluated in-bag at a FIXED threshold (3.30); RQ62's BCa CI is
bias-corrected + accelerated and evaluated OOB at RE-CALIBRATED ensemble
thresholds. The comparison is therefore directional (does the ensemble +
BCa + OOB keep the interval within the original-rule width) rather than a pure
like-for-like CI-method swap; this mirrors RQ54's H54b and RQ59's H59c caveat.

This script is pure reanalysis (numpy + stdlib only; scipy / sklearn / Whisper
are NOT required). The KL calibration is imported verbatim from
``src.llm_semantic_critic`` (RQ34/RQ58); the lang-id calibration, bootstrap
index draw, and OOB protocol mirror RQ44 (PR #963); the BCa CI helpers (norm_cdf,
norm_ppf, bca_ci) and the jackknife acceleration structure are imported verbatim
from RQ59 (PR #980); the mode counter is imported verbatim from RQ48 (PR #965).
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
RQ43_JSON = (
    PROJECT_ROOT
    / "results"
    / "frontier"
    / "three_tier_cascade"
    / "three_tier_cascade_results.json"
)
RQ58_JSON = (
    PROJECT_ROOT
    / "results"
    / "frontier"
    / "kl_corrected_router"
    / "kl_corrected_router_results.json"
)
AISHELL4_JSON = (
    PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)
RQ46_JSON = (
    PROJECT_ROOT
    / "results"
    / "frontier"
    / "bootstrap_pareto"
    / "bootstrap_pareto_results.json"
)
RQ54_JSON = (
    PROJECT_ROOT
    / "results"
    / "frontier"
    / "cascade_f1_calibration"
    / "cascade_f1_results.json"
)
RQ59_JSON = (
    PROJECT_ROOT
    / "results"
    / "frontier"
    / "cascade_youdens_j"
    / "cascade_youdens_j_results.json"
)
OUT_DIR = PROJECT_ROOT / "results" / "frontier" / "ensemble_cascade_gate"
OUT_JSON = OUT_DIR / "ensemble_cascade_results.json"
OUT_CSV = OUT_DIR / "ensemble_cascade_results.csv"

# ------------------------------------------ import RQ59's BCa + RQ48's count_modes + RQ44's lang-id cal
# RQ59 (PR #980) contributes the BCa CI scaffolding (norm_cdf, norm_ppf, bca_ci)
# and the jackknife acceleration structure; reusing them guarantees the CI
# methodology is byte-identical to RQ59 (only the gate changes: ensemble here
# vs single-KL in RQ59). RQ48 (PR #965) contributes count_modes. RQ44 (PR #963)
# contributes the lang-id calibration rule, the THRESHOLD_GRID, EPS, and the
# bootstrap index draw convention. RQ58 / src.llm_semantic_critic contributes
# the KL calibration rule (calibrate_threshold_at_specificity).
_RQ59_DIR = PROJECT_ROOT / "results" / "frontier" / "cascade_youdens_j"
_RQ48_DIR = PROJECT_ROOT / "results" / "frontier" / "calibration_rule_comparison"
_RQ44_DIR = PROJECT_ROOT / "results" / "frontier" / "bootstrap_threshold_stability"
sys.path.insert(0, str(_RQ59_DIR))
sys.path.insert(0, str(_RQ48_DIR))
sys.path.insert(0, str(_RQ44_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import cascade_youdens_j_analysis as rq59  # noqa: E402  (path-injected import)
import calibration_rule_analysis as rq48  # noqa: E402  (path-injected import)
import bootstrap_threshold_analysis as rq44  # noqa: E402  (path-injected import)
from src.llm_semantic_critic import (  # noqa: E402
    calibrate_threshold_at_specificity as _calibrate_kl_threshold,
)

# ------------------------------------------------------------------ constants
# RQ59's BCa CI helpers and RQ48's mode counter, re-exported for traceability.
norm_cdf = rq59.norm_cdf       # standard-normal forward CDF (erfc-based, no scipy)
norm_ppf = rq59.norm_ppf       # standard-normal inverse CDF (Acklam + Halley)
bca_ci = rq59.bca_ci           # BCa 95% CI (bias-corrected + accelerated)
count_modes = rq48.count_modes  # RQ48's >= 5% mode counter
calibrate_lang_threshold = rq44.calibrate_threshold_at_spec  # RQ44's lang-id cal
LANG_ID_GRID = rq44.THRESHOLD_GRID  # 0.00, 0.01, ..., 2.00 (201 pts)
EPS = rq44.EPS                          # 1e-9

N_BOOT = 10000            # task-specified bootstrap iterations
SEED = 42                 # task-specified seed
MIN_MODE_FRACTION = 0.05  # "mode" = distinct threshold with >= 5% frequency (RQ48)
ALPHA = 0.05              # 95% CI

COMPUTE_TINY = 1.0        # whisper-tiny relative compute (RQ43)
COMPUTE_BASE = 1.93       # whisper-base relative compute (RQ43 / runtime_cascade)
CATASTROPHIC_CPWER = 1.0  # cpWER > 1.0 => hallucination label (RQ44/RQ48/RQ54/RQ59)
TARGET_SPECIFICITY = 0.90  # both KL and lang-id calibrated at >= 90% specificity

# Fixed in-sample thresholds (the calibrated values on the full 77 windows).
# KL_THRESHOLD = 5.418144 (RQ58's 2-gram KL at >=90% specificity); the task
# rounds to 5.42. We use the exact calibrated value (not the rounded 5.42) so
# the in-sample calibration reproduces RQ58's threshold bit-for-bit.
KL_THRESHOLD_IN_SAMPLE = 5.418144   # RQ58's empirically-calibrated 2-gram KL
LANG_ID_THRESHOLD_IN_SAMPLE = 0.38  # RQ44's >=90%-specificity lang-id threshold

# RQ43 / RQ46 / RQ54 / RQ59 anchors (the controlled-comparison reference values).
RQ43_KL_THRESHOLD = 3.30             # RQ43's original n=3 KL threshold (kl_sep)
RQ43_CASCADE_CPWER = 0.888947        # RQ43 in-sample cascade cpWER @ KL=3.30
RQ43_BASELINE_CPWER = 1.590909       # always-tiny-separated
RQ43_BASE_RATIO = 0.428031           # model_scale separated base/tiny CER ratio
RQ46_CI_LO = 0.767399                # RQ46 bootstrap percentile CI lo @ KL=3.30
RQ46_CI_HI = 1.016343                # RQ46 bootstrap percentile CI hi @ KL=3.30
RQ46_CI_WIDTH = round(RQ46_CI_HI - RQ46_CI_LO, 4)  # 0.2489 (H62c anchor)
RQ54_F1_ESCALATION = 0.831169        # RQ54's F1 in-sample escalation fraction
RQ54_F1_OOB_MEDIAN_CPWER = 0.779853  # RQ54's F1 OOB median cpWER
RQ54_F1_BCA_WIDTH = 0.248072         # RQ54's F1 BCa CI width
RQ59_YOUDENS_J_ESCALATION = 0.831169  # RQ59's Youden's J escalation (== RQ54)
RQ59_YOUDENS_J_OOB_MEDIAN_CPWER = 0.782  # RQ59's Youden's J OOB median cpWER
RQ58_KL_THRESHOLD = 5.418144         # RQ58's 2-gram KL at >=90% specificity
RQ44_LANG_ID_THRESHOLD = 0.38        # RQ44's lang-id at >=90% specificity

# Hypothesis kill thresholds.
H62A_MAX_ESCALATION = 0.831   # ensemble: kill if escalation fraction >= 0.831
H62B_MAX_CPWER = 0.889        # ensemble: kill if OOB median cpwer > 0.889
H62C_MAX_WIDTH = RQ46_CI_WIDTH  # BCa cpWER CI width: kill if > 0.2489


# --------------------------------------------------------------- data loading
def load_cascade_windows() -> dict[str, Any]:
    """Load RQ43's cascade data + RQ58's KL/lang-id scores, joined by window_id.

    Reads ``three_tier_cascade_results.json`` ``per_window`` for the cascade
    cpWER corpus (tiny_sep_cpwer, base_sep_cpwer, kl_sep) and
    ``kl_corrected_router_results.json`` ``per_window`` for the ensemble
    detector signals (kl_score = RQ58's 2-gram KL, lang_id_entropy = RQ13's
    max-across-speakers script-category entropy). The two are joined by
    window_id (both are 0..76 in order, verified in tests). Returns a dict of
    float arrays and asserts n == 77.
    """
    rq43_data = json.loads(RQ43_JSON.read_text(encoding="utf-8"))
    pw43 = rq43_data["per_window"]
    assert len(pw43) == 77, f"expected 77 RQ43 windows, got {len(pw43)}"

    rq58_data = json.loads(RQ58_JSON.read_text(encoding="utf-8"))
    pw58 = rq58_data["per_window"]
    assert len(pw58) == 77, f"expected 77 RQ58 windows, got {len(pw58)}"

    by_id58 = {w["window_id"]: w for w in pw58}
    window_ids = [w["window_id"] for w in pw43]
    assert window_ids == [w["window_id"] for w in pw58], (
        "RQ43 and RQ58 window_id order mismatch; cannot join by position")

    tiny = np.array([float(w["tiny_sep_cpwer"]) for w in pw43], dtype=float)
    base = np.array([float(w["base_sep_cpwer"]) for w in pw43], dtype=float)
    kl_sep = np.array([float(w["kl_sep"]) for w in pw43], dtype=float)
    kl58 = np.array(
        [float(by_id58[wid]["kl_score"]) for wid in window_ids], dtype=float)
    lang = np.array(
        [float(by_id58[wid]["lang_id_entropy"]) for wid in window_ids],
        dtype=float)
    return {
        "tiny": tiny, "base": base, "kl_sep": kl_sep,
        "kl": kl58, "lang": lang, "window_id": window_ids,
    }


# --------------------------------------------------------------- KL calibration
def calibrate_kl_threshold(
    kl: np.ndarray, labels: np.ndarray,
    target_spec: float = TARGET_SPECIFICITY,
) -> dict[str, Any]:
    """RQ58's KL threshold calibration (candidate-set, >=90% specificity).

    Wraps ``src.llm_semantic_critic.calibrate_threshold_at_specificity`` so the
    in-sample threshold reproduces RQ58's 5.418144 bit-for-bit. Candidate
    thresholds are drawn from the union of in-bag neg/pos KL scores; the
    SMALLEST candidate with fp <= max_fp (= floor((1-target_spec)*n_neg)) is
    chosen (highest sensitivity at the specificity floor). Returns a dict with
    ``threshold``, ``specificity``, ``n_neg``, ``max_fp`` (RQ58's shape)."""
    kl = np.asarray(kl, dtype=float)
    labels = np.asarray(labels, dtype=int)
    neg = kl[labels == 0].tolist()
    pos = kl[labels == 1].tolist()
    return _calibrate_kl_threshold(neg, pos, target_spec=target_spec)


# --------------------------------------------------------------- ensemble gate
def escalate_mask(
    kl: np.ndarray, lang: np.ndarray, kl_thr: float, lang_thr: float,
    gate: str = "or", eps: float = EPS,
) -> np.ndarray:
    """Boolean escalation mask for the ensemble gate.

    ``gate = "or"``  -> escalate if KL >= kl_thr OR lang_id >= lang_thr.
    ``gate = "and"`` -> escalate if KL >= kl_thr AND lang_id >= lang_thr.
    Uses the ``>= - eps`` flagging convention (RQ44/RQ48/RQ54/RQ59). ``kl_thr``
    or ``lang_thr`` of +inf correctly flags nothing (no finite score >= inf)."""
    kl = np.asarray(kl, dtype=float)
    lang = np.asarray(lang, dtype=float)
    kl_flag = kl >= kl_thr - eps
    lang_flag = lang >= lang_thr - eps
    if gate == "or":
        return kl_flag | lang_flag
    if gate == "and":
        return kl_flag & lang_flag
    raise ValueError(f"unknown gate {gate!r}; expected 'or' or 'and'")


def cascade_cpwer_at_thresholds(
    tiny: np.ndarray, base: np.ndarray, kl: np.ndarray, lang: np.ndarray,
    kl_thr: float, lang_thr: float, gate: str = "or",
) -> float:
    """In-sample ensemble cascade cpWER at ``(kl_thr, lang_thr)``.

    Escalation: ensemble gate (OR/AND) flags the window -> base; else tiny.
    Returns the mean selected cpWER. Returns 0.0 for empty input."""
    tiny = np.asarray(tiny, dtype=float)
    base = np.asarray(base, dtype=float)
    if tiny.size == 0:
        return 0.0
    esc = escalate_mask(kl, lang, kl_thr, lang_thr, gate=gate)
    selected = np.where(esc, base, tiny)
    return float(selected.mean())


def cascade_compute_at_thresholds(
    kl: np.ndarray, lang: np.ndarray, kl_thr: float, lang_thr: float,
    gate: str = "or",
) -> float:
    """Cascade compute = 1.0*(1-f) + 1.93*f, f = escalation fraction.

    The ensemble gate cost is negligible and folded into the 1.0x tiny budget
    (RQ43 convention)."""
    kl = np.asarray(kl, dtype=float)
    lang = np.asarray(lang, dtype=float)
    if kl.size == 0:
        return 0.0
    frac = float(np.mean(escalate_mask(kl, lang, kl_thr, lang_thr, gate=gate)))
    return COMPUTE_TINY * (1.0 - frac) + COMPUTE_BASE * frac


def cascade_oob_cpwer(
    tiny: np.ndarray, base: np.ndarray, kl: np.ndarray, lang: np.ndarray,
    kl_thr: float, lang_thr: float, in_bag_idx: np.ndarray, gate: str = "or",
) -> dict[str, Any]:
    """Ensemble cascade cpWER on the out-of-bag (OOB) windows.

    Mirrors RQ44's / RQ54's / RQ59's OOB protocol but routes escalated windows
    to ``base`` (whisper-base) and non-escalated to ``tiny`` (whisper-tiny)
    via the ensemble gate. Returns the mean selected cpWER over the OOB windows
    (``nan`` if there are none), the OOB size, and the escalation count."""
    n = len(kl)
    all_idx = np.arange(n)
    in_bag_set = np.unique(np.asarray(in_bag_idx, dtype=int))
    oob_mask = ~np.isin(all_idx, in_bag_set)
    n_oob = int(oob_mask.sum())
    if n_oob == 0:
        return {"cpwer": float("nan"), "n_oob": 0, "n_escalated": 0}
    esc = escalate_mask(
        kl[oob_mask], lang[oob_mask], kl_thr, lang_thr, gate=gate)
    selected = np.where(esc, base[oob_mask], tiny[oob_mask])
    return {"cpwer": float(selected.mean()), "n_oob": n_oob,
            "n_escalated": int(esc.sum())}


# --------------------------------------------------------------- bootstrap ensemble cascade
def bootstrap_ensemble_cascade(
    tiny: np.ndarray, base: np.ndarray, kl: np.ndarray, lang: np.ndarray,
    labels: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Bootstrap the ensemble cascade over ``n_boot`` resamples.

    For each resample: draw n indices with replacement, re-calibrate the KL
    threshold (RQ58's candidate-set rule) AND the lang-id threshold (RQ44's
    grid rule) on the in-bag windows, and evaluate the ensemble cascade cpWER
    (OR and AND gates) on the OOB windows. Both gates share the same
    resample indices and calibrated thresholds so the OR-vs-AND comparison is
    paired.

    Returns a dict with:
      ``boot_idx``            -- (n_boot, n) int array of resample indices
      ``kl_thresholds``       -- (n_boot,) calibrated KL threshold per resample
      ``lang_thresholds``     -- (n_boot,) calibrated lang-id threshold per resample
      ``oob_cpwer_or``        -- (n_boot,) OOB cascade cpWER, OR gate (nan if empty)
      ``oob_cpwer_and``       -- (n_boot,) OOB cascade cpWER, AND gate (nan if empty)
      ``n_oob``               -- (n_boot,) OOB size per resample
      ``n_escalated_oob_or``  -- (n_boot,) escalated count within OOB, OR gate
      ``n_escalated_oob_and`` -- (n_boot,) escalated count within OOB, AND gate
    """
    tiny = np.asarray(tiny, dtype=float)
    base = np.asarray(base, dtype=float)
    kl = np.asarray(kl, dtype=float)
    lang = np.asarray(lang, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = kl.shape[0]

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))  # (B, n)

    kl_thresholds = np.empty(n_boot, dtype=float)
    lang_thresholds = np.empty(n_boot, dtype=float)
    oob_cpwer_or = np.empty(n_boot, dtype=float)
    oob_cpwer_and = np.empty(n_boot, dtype=float)
    n_oob_arr = np.empty(n_boot, dtype=int)
    n_esc_or = np.empty(n_boot, dtype=int)
    n_esc_and = np.empty(n_boot, dtype=int)

    for b in range(n_boot):
        idx = boot_idx[b]
        kl_in = kl[idx]
        lang_in = lang[idx]
        labels_in = labels[idx]

        # --- re-calibrate KL threshold (RQ58 candidate-set rule) on in-bag
        kl_cal = calibrate_kl_threshold(kl_in, labels_in, TARGET_SPECIFICITY)
        kl_thr = float(kl_cal["threshold"])

        # --- re-calibrate lang-id threshold (RQ44 grid rule) on in-bag
        lang_cal = calibrate_lang_threshold(
            lang_in, labels_in, grid=LANG_ID_GRID, target_spec=TARGET_SPECIFICITY)
        lang_thr = float(lang_cal["threshold"])

        kl_thresholds[b] = kl_thr
        lang_thresholds[b] = lang_thr

        # --- OOB evaluation (both gates share the same thresholds/resample)
        counts = np.bincount(idx, minlength=n)
        oob_mask = counts == 0
        no = int(oob_mask.sum())
        n_oob_arr[b] = no
        if no == 0:
            oob_cpwer_or[b] = float("nan")
            oob_cpwer_and[b] = float("nan")
            n_esc_or[b] = 0
            n_esc_and[b] = 0
            continue

        oob_kl = kl[oob_mask]
        oob_lang = lang[oob_mask]
        oob_tiny = tiny[oob_mask]
        oob_base = base[oob_mask]

        kl_flag = oob_kl >= kl_thr - EPS
        lang_flag = oob_lang >= lang_thr - EPS
        esc_or = kl_flag | lang_flag
        esc_and = kl_flag & lang_flag

        oob_cpwer_or[b] = float(np.where(esc_or, oob_base, oob_tiny).mean())
        oob_cpwer_and[b] = float(np.where(esc_and, oob_base, oob_tiny).mean())
        n_esc_or[b] = int(esc_or.sum())
        n_esc_and[b] = int(esc_and.sum())

    return {
        "boot_idx": boot_idx,
        "kl_thresholds": kl_thresholds,
        "lang_thresholds": lang_thresholds,
        "oob_cpwer_or": oob_cpwer_or,
        "oob_cpwer_and": oob_cpwer_and,
        "n_oob": n_oob_arr,
        "n_escalated_oob_or": n_esc_or,
        "n_escalated_oob_and": n_esc_and,
    }


# --------------------------------------------------------------- jackknife acceleration
def jackknife_acceleration(
    tiny: np.ndarray, base: np.ndarray, kl: np.ndarray, lang: np.ndarray,
    labels: np.ndarray, gate: str = "or",
) -> tuple[float, np.ndarray]:
    """Delete-1 jackknife acceleration for the BCa CI (ensemble cascade).

    For each i in 0..n-1: leave window i out, re-calibrate BOTH thresholds on
    the remaining n-1 windows, and compute the in-sample ensemble cascade
    cpWER on those n-1 windows (theta_(i)). The acceleration is

        a = sum( (theta_bar - theta_(i))^3 ) / ( 6 * sum( (theta_bar - theta_(i))^2 )^1.5 )

    Returns (a, theta_loo). a = 0.0 when the denominator is 0 (no variation),
    which collapses BCa to the bias-corrected percentile. Mirrors RQ59's
    jackknife structure exactly; only the gate changes (ensemble vs single-KL).
    """
    tiny = np.asarray(tiny, dtype=float)
    base = np.asarray(base, dtype=float)
    kl = np.asarray(kl, dtype=float)
    lang = np.asarray(lang, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = kl.shape[0]
    theta_loo = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        kl_sub = kl[mask]
        lang_sub = lang[mask]
        labels_sub = labels[mask]
        kl_cal = calibrate_kl_threshold(kl_sub, labels_sub, TARGET_SPECIFICITY)
        lang_cal = calibrate_lang_threshold(
            lang_sub, labels_sub, grid=LANG_ID_GRID, target_spec=TARGET_SPECIFICITY)
        theta_loo[i] = cascade_cpwer_at_thresholds(
            tiny[mask], base[mask], kl_sub, lang_sub,
            float(kl_cal["threshold"]), float(lang_cal["threshold"]), gate=gate)
    theta_bar = float(theta_loo.mean())
    diff = theta_bar - theta_loo
    scale = max(abs(theta_bar), 1.0)
    if float(np.max(np.abs(diff))) < 1e-12 * scale:
        return 0.0, theta_loo
    num = float(np.sum(diff ** 3))
    den = 6.0 * (float(np.sum(diff ** 2)) ** 1.5)
    a = num / den if den > 0 else 0.0
    return a, theta_loo


# --------------------------------------------------------------- finite-only stats
def _finite_stats(arr: np.ndarray) -> dict[str, float]:
    """Descriptive stats on the finite subset of ``arr`` (inf/nan dropped).

    The KL threshold distribution can contain ``+inf`` (resamples where the
    in-bag set has too few negatives to meet the >=90% specificity floor, so
    no finite threshold flags anything). Those resamples are legitimate
    operating points (no KL escalation) and are counted by ``count_modes``,
    but mean/std/min/max are not meaningful with inf mixed in. This helper
    reports the descriptive stats on the finite subset plus the inf fraction."""
    arr = np.asarray(arr, dtype=float)
    n_total = int(arr.size)
    finite = arr[np.isfinite(arr)]
    n_finite = int(finite.size)
    n_inf = int(np.sum(np.isposinf(arr)))
    if n_finite == 0:
        return {"median": float("nan"), "mean": float("nan"),
                "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_finite": 0, "n_inf": n_inf,
                "n_total": n_total, "inf_fraction": round(n_inf / n_total, 6)
                if n_total > 0 else 0.0}
    return {
        "median": round(float(np.median(finite)), 6),
        "mean": round(float(np.mean(finite)), 6),
        "std": round(float(np.std(finite)), 6),
        "min": round(float(np.min(finite)), 6),
        "max": round(float(np.max(finite)), 6),
        "n_finite": n_finite, "n_inf": n_inf, "n_total": n_total,
        "inf_fraction": round(n_inf / n_total, 6) if n_total > 0 else 0.0,
    }


# --------------------------------------------------------------- CSV output
def write_bootstrap_csv(
    path: Path, boot_idx: np.ndarray, kl_thresholds: np.ndarray,
    lang_thresholds: np.ndarray, oob_cpwer_or: np.ndarray,
    oob_cpwer_and: np.ndarray, n_oob: np.ndarray,
    n_escalated_oob_or: np.ndarray, n_escalated_oob_and: np.ndarray,
    n_windows: int,
) -> None:
    """Write the per-resample bootstrap table as CSV.

    One row per bootstrap resample (B rows) plus a header. Columns:
    ``resample, kl_threshold, lang_threshold, oob_cpwer_or, oob_cpwer_and,
    n_oob, n_escalated_oob_or, n_escalated_oob_and, oob_fraction,
    escalation_fraction_oob_or, escalation_fraction_oob_and``. OOB cpWER is
    blank when the OOB set was empty (nan)."""
    B = int(kl_thresholds.shape[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "resample", "kl_threshold", "lang_threshold",
            "oob_cpwer_or", "oob_cpwer_and", "n_oob",
            "n_escalated_oob_or", "n_escalated_oob_and", "oob_fraction",
            "escalation_fraction_oob_or", "escalation_fraction_oob_and",
        ])
        for b in range(B):
            no = int(n_oob[b])
            cp_or = float(oob_cpwer_or[b])
            cp_and = float(oob_cpwer_and[b])
            esc_or = int(n_escalated_oob_or[b])
            esc_and = int(n_escalated_oob_and[b])
            w.writerow([
                b,
                round(float(kl_thresholds[b]), 6),
                round(float(lang_thresholds[b]), 6),
                "" if (no == 0 or math.isnan(cp_or)) else round(cp_or, 6),
                "" if (no == 0 or math.isnan(cp_and)) else round(cp_and, 6),
                no,
                esc_or,
                esc_and,
                round(no / n_windows, 6) if n_windows > 0 else 0.0,
                round(esc_or / no, 6) if no > 0 else "",
                round(esc_and / no, 6) if no > 0 else "",
            ])


# --------------------------------------------------------------- driver
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = load_cascade_windows()
    tiny = windows["tiny"]
    base = windows["base"]
    kl_sep = windows["kl_sep"]    # RQ43's n=3 KL (for the anchor reproduction)
    kl = windows["kl"]            # RQ58's 2-gram KL (the ensemble detector)
    lang = windows["lang"]        # RQ13's lang-id entropy (the ensemble detector)
    n = kl.shape[0]
    labels = (tiny > CATASTROPHIC_CPWER).astype(int)  # 1 = hallucinated
    n_hall = int(labels.sum())
    n_clean = int((labels == 0).sum())

    # --- controlled-comparison smoke: RQ43's original rule reproduces 0.888947
    # (uses RQ43's n=3 kl_sep at threshold 3.30, NOT the ensemble KL)
    rq43_cas = float(np.where(kl_sep >= RQ43_KL_THRESHOLD, base, tiny).mean())
    baseline = float(tiny.mean())
    assert abs(rq43_cas - RQ43_CASCADE_CPWER) < 1e-4, (
        f"RQ43 cascade @ KL=3.30 = {rq43_cas}, expected ~{RQ43_CASCADE_CPWER}")
    assert abs(baseline - RQ43_BASELINE_CPWER) < 1e-4
    assert n_hall == 37 and n_clean == 40, f"label counts {n_hall}/{n_clean}"
    # base ratio is constant per window (0.428031)
    ratio = base / tiny
    assert np.allclose(ratio, RQ43_BASE_RATIO, atol=1e-4), (
        f"base/tiny ratio not constant {RQ43_BASE_RATIO}: min={ratio.min()}, "
        f"max={ratio.max()}")

    # --- in-sample calibration (full 77 windows): KL (RQ58) + lang-id (RQ44)
    kl_cal = calibrate_kl_threshold(kl, labels, TARGET_SPECIFICITY)
    lang_cal = calibrate_lang_threshold(
        lang, labels, grid=LANG_ID_GRID, target_spec=TARGET_SPECIFICITY)
    kl_thr_in = float(kl_cal["threshold"])
    lang_thr_in = float(lang_cal["threshold"])
    # verify the in-sample thresholds reproduce RQ58 / RQ44
    assert abs(kl_thr_in - KL_THRESHOLD_IN_SAMPLE) < 1e-4, (
        f"KL threshold {kl_thr_in} != RQ58's {KL_THRESHOLD_IN_SAMPLE}")
    assert abs(lang_thr_in - LANG_ID_THRESHOLD_IN_SAMPLE) < 1e-4, (
        f"lang-id threshold {lang_thr_in} != RQ44's {LANG_ID_THRESHOLD_IN_SAMPLE}")

    # --- in-sample ensemble gate (OR and AND) at the calibrated thresholds
    theta_hat_or = cascade_cpwer_at_thresholds(
        tiny, base, kl, lang, kl_thr_in, lang_thr_in, gate="or")
    theta_hat_and = cascade_cpwer_at_thresholds(
        tiny, base, kl, lang, kl_thr_in, lang_thr_in, gate="and")
    compute_or = cascade_compute_at_thresholds(
        kl, lang, kl_thr_in, lang_thr_in, gate="or")
    compute_and = cascade_compute_at_thresholds(
        kl, lang, kl_thr_in, lang_thr_in, gate="and")
    frac_or = float(np.mean(escalate_mask(kl, lang, kl_thr_in, lang_thr_in, "or")))
    frac_and = float(np.mean(escalate_mask(kl, lang, kl_thr_in, lang_thr_in, "and")))
    n_esc_or = int(escalate_mask(kl, lang, kl_thr_in, lang_thr_in, "or").sum())
    n_esc_and = int(escalate_mask(kl, lang, kl_thr_in, lang_thr_in, "and").sum())

    # single-detector in-sample escalation (for the comparison table)
    n_esc_kl_only = int(np.sum(kl >= kl_thr_in - EPS))
    n_esc_lang_only = int(np.sum(lang >= lang_thr_in - EPS))
    frac_kl_only = n_esc_kl_only / n
    frac_lang_only = n_esc_lang_only / n

    # --- bootstrap ensemble cascade (B=10000, seed=42) -- both gates paired
    boot = bootstrap_ensemble_cascade(
        tiny, base, kl, lang, labels, n_boot=N_BOOT, seed=SEED)
    boot_kl_thr = boot["kl_thresholds"]
    boot_lang_thr = boot["lang_thresholds"]
    boot_oob_or = boot["oob_cpwer_or"]
    boot_oob_and = boot["oob_cpwer_and"]
    n_oob_mean = float(np.mean(boot["n_oob"]))

    # --- mode count on the KL-threshold and lang-id-threshold distributions
    kl_modes = count_modes(boot_kl_thr, MIN_MODE_FRACTION)
    lang_modes = count_modes(boot_lang_thr, MIN_MODE_FRACTION)

    # --- jackknife acceleration + BCa CI for OR and AND
    accel_or, theta_loo_or = jackknife_acceleration(
        tiny, base, kl, lang, labels, gate="or")
    accel_and, theta_loo_and = jackknife_acceleration(
        tiny, base, kl, lang, labels, gate="and")
    bca_or = bca_ci(theta_hat_or, boot_oob_or, accel_or, alpha=ALPHA)
    bca_and = bca_ci(theta_hat_and, boot_oob_and, accel_and, alpha=ALPHA)
    bca_width_or = bca_or["hi"] - bca_or["lo"]
    bca_width_and = bca_and["hi"] - bca_and["lo"]
    oob_median_or = bca_or["median"]
    oob_median_and = bca_and["median"]
    oob_mean_or = float(np.nanmean(boot_oob_or))
    oob_mean_and = float(np.nanmean(boot_oob_and))

    # --- hypothesis verdicts (primary on OR; AND reported as secondary)
    h62a_supported_or = frac_or < H62A_MAX_ESCALATION
    h62a_supported_and = frac_and < H62A_MAX_ESCALATION
    h62b_supported_or = oob_median_or <= H62B_MAX_CPWER
    h62b_supported_and = oob_median_and <= H62B_MAX_CPWER
    h62c_supported_or = bca_width_or <= H62C_MAX_WIDTH
    h62c_supported_and = bca_width_and <= H62C_MAX_WIDTH

    # --- RQ46 original-rule reference (read from RQ46 JSON @ KL=3.30)
    rq46_ref = {"ci_lo": RQ46_CI_LO, "ci_hi": RQ46_CI_HI, "width": RQ46_CI_WIDTH}
    try:
        rq46_data = json.loads(RQ46_JSON.read_text(encoding="utf-8"))
        for p in rq46_data.get("per_point", []):
            if abs(float(p["threshold"]) - RQ43_KL_THRESHOLD) < 1e-9:
                rq46_ref = {
                    "ci_lo": float(p["cpwer_ci_lo"]),
                    "ci_hi": float(p["cpwer_ci_hi"]),
                    "width": round(
                        float(p["cpwer_ci_hi"]) - float(p["cpwer_ci_lo"]), 6)}
                break
    except (OSError, ValueError, KeyError) as exc:
        print(f"[warn] RQ46 reference JSON unreadable ({exc}); "
              f"falling back to hardcoded RQ46 anchors.", file=sys.stderr)

    # --- RQ59 Youden's J reference (read from RQ59 JSON)
    rq59_ref = {
        "youdens_j_threshold": None,
        "youdens_j_escalation": RQ59_YOUDENS_J_ESCALATION,
        "youdens_j_oob_median_cpwer": RQ59_YOUDENS_J_OOB_MEDIAN_CPWER,
    }
    try:
        rq59_data = json.loads(RQ59_JSON.read_text(encoding="utf-8"))
        rq59_ref["youdens_j_threshold"] = float(
            rq59_data["in_sample_youdens_j_calibration"]["threshold"])
        rq59_ref["youdens_j_escalation"] = float(
            rq59_data["in_sample_youdens_j_calibration"]["escalation_fraction"])
        rq59_ref["youdens_j_oob_median_cpwer"] = float(
            rq59_data["bootstrap_oob_cpwer_distribution"]["median"])
        rq59_ref["youdens_j_bca_width"] = float(rq59_data["bca_ci"]["width"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"[warn] RQ59 reference JSON unreadable ({exc}); "
              f"falling back to hardcoded RQ59 anchors.", file=sys.stderr)

    # --- RQ54 F1 reference (read from RQ54 JSON)
    rq54_ref = {
        "f1_threshold": None, "f1_escalation": RQ54_F1_ESCALATION,
        "f1_oob_median_cpwer": RQ54_F1_OOB_MEDIAN_CPWER,
        "f1_bca_width": RQ54_F1_BCA_WIDTH,
    }
    try:
        rq54_data = json.loads(RQ54_JSON.read_text(encoding="utf-8"))
        rq54_ref["f1_threshold"] = float(
            rq54_data["in_sample_f1_calibration"]["threshold"])
        rq54_ref["f1_escalation"] = float(
            rq54_data["in_sample_f1_calibration"]["escalation_fraction"])
        rq54_ref["f1_oob_median_cpwer"] = float(
            rq54_data["bootstrap_oob_cpwer_distribution"]["median"])
        rq54_ref["f1_bca_width"] = float(rq54_data["bca_ci"]["width"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"[warn] RQ54 reference JSON unreadable ({exc}); "
              f"falling back to hardcoded RQ54 anchors.", file=sys.stderr)

    summary: dict[str, Any] = {
        "label": "experimental/frontier",
        "rq": ("RQ62: Cascade with KL+lang-id ensemble gate -- does a "
               "two-detector ensemble produce a less aggressive or more "
               "robust cascade operating point than KL alone?"),
        "builds_on": {
            "RQ13": "results/frontier/diverse_hallucination_detector/ (PR #904, lang-id entropy detector)",
            "RQ43": "results/frontier/three_tier_cascade/ (PR #959, 3-tier KL cascade)",
            "RQ44": "results/frontier/bootstrap_threshold_stability/ (PR #963, OOB bootstrap + lang-id cal)",
            "RQ46": "results/frontier/bootstrap_pareto/ (PR #966, original-rule CI anchor)",
            "RQ48": "results/frontier/calibration_rule_comparison/ (PR #965, count_modes)",
            "RQ54": "results/frontier/cascade_f1_calibration/ (PR #971, F1 cascade comparison)",
            "RQ58": "results/frontier/kl_corrected_router/ (PR #975, 2-gram KL detector + cal)",
            "RQ59": "results/frontier/cascade_youdens_j/ (PR #980, BCa CI + jackknife framework)",
        },
        "source_data": {
            "rq43_json": str(RQ43_JSON.relative_to(PROJECT_ROOT)),
            "rq43_label": "experimental/frontier",
            "rq58_json": str(RQ58_JSON.relative_to(PROJECT_ROOT)),
            "rq58_label": "experimental/frontier",
            "aishell4_json": str(AISHELL4_JSON.relative_to(PROJECT_ROOT)),
            "aishell4_label": "external/sanity-check",
            "aishell4_asr_model": "whisper-tiny",
        },
        "method": (
            "REANALYSIS (no ASR run). Loads RQ43's 77 per-window cascade data "
            "(tiny_sep_cpwer, base_sep_cpwer, kl_sep) and RQ58's 77 per-window "
            "ensemble detector signals (kl_score = 2-gram character KL, "
            "lang_id_entropy = RQ13 max-across-speakers script-category "
            "entropy), joined by window_id so the cascade corpus is "
            "byte-identical to RQ43/RQ54/RQ59. The ONLY change vs RQ59 is the "
            "gate: RQ59's single-KL Youden's J gate is replaced by a KL+lang-id "
            "ENSEMBLE gate (OR: escalate if either flags; AND: escalate if both "
            "flag). KL threshold calibrated via RQ58's candidate-set "
            ">=90%-specificity rule (reproduces 5.418144); lang-id threshold "
            "via RQ44's grid >=90%-specificity rule (reproduces 0.38). "
            "Hallucination label = tiny_sep_cpwer > 1.0 (37 hall / 40 clean). "
            "Cascade: tiny on all windows -> ensemble gate -> base "
            "(cpWER = tiny * 0.428031, RQ43's separated ratio). Bootstrap "
            "B=10000 seed=42: per resample re-calibrate BOTH thresholds on "
            "in-bag windows, evaluate ensemble cascade cpWER (OR and AND) on "
            "OOB windows (RQ44 OOB protocol). BCa 95% CI on the OOB cpWER "
            "distribution (bias-correction z0 from the in-sample point estimate "
            "theta_hat; acceleration from a delete-1 jackknife, separately for "
            "OR and AND). Mode count via RQ48's count_modes (>= 5% frequency) "
            "on both the KL-threshold and lang-id-threshold distributions."
        ),
        "controlled_comparison_note": (
            "The cascade simulation is held fixed at RQ43's actual "
            "implementation (real whisper-tiny cpWER for tier 1; base cpWER = "
            "tiny * 0.428031 for tier 3) so the H62b comparison to RQ43's "
            "0.888947 anchor, the H62a comparison to RQ54/RQ59's 83.1% "
            "escalation, and the H62c comparison to RQ46's 0.2489 width are "
            "apples-to-apples. The ONLY independent variable vs RQ59 is the "
            "gate (ensemble KL+lang-id instead of single-KL Youden's J)."
        ),
        "n_windows": n,
        "n_hallucinated": n_hall,
        "n_clean": n_clean,
        "hallucination_label_rule": "tiny_sep_cpwer > 1.0 (== always_separated_cpwer > 1.0)",
        "ensemble_config": {
            "gate_logic": "OR (primary): escalate if KL >= kl_thr OR lang_id >= lang_thr; "
                          "AND (secondary): escalate if KL >= kl_thr AND lang_id >= lang_thr",
            "kl_detector": {
                "source": "RQ58 (PR #975), 2-gram character KL divergence",
                "threshold_in_sample": round(kl_thr_in, 6),
                "threshold_rounded": 5.42,
                "calibration_rule": "candidate-set, >=90% specificity, smallest-threshold tie-break (RQ58)",
                "kl_range": [round(float(kl.min()), 6), round(float(kl.max()), 6)],
            },
            "lang_id_detector": {
                "source": "RQ13 (PR #904), max-across-speakers script-category entropy",
                "threshold_in_sample": round(lang_thr_in, 6),
                "threshold_rounded": 0.38,
                "calibration_rule": "0.01-step grid [0.00, 2.00], >=90% specificity, "
                                    "max sensitivity, tie-break higher spec then lower threshold (RQ44)",
                "lang_id_range": [round(float(lang.min()), 6), round(float(lang.max()), 6)],
            },
        },
        "compute_model": {"tiny": COMPUTE_TINY, "base": COMPUTE_BASE,
                          "source": "RQ43 / runtime_cascade (base 1.93x slower)"},
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "resample_size": n,
                      "oob_protocol": "RQ44 out_of_bag (calibrate in-bag, evaluate OOB)",
                      "expected_oob_size": round(n * ((1 - 1 / n) ** n), 4),
                      "mean_oob_size": round(n_oob_mean, 4),
                      "paired_or_and": True},
        "bca_method": {
            "theta_hat": "in-sample ensemble cascade cpWER at the in-sample calibrated thresholds",
            "boot_samples": "OOB ensemble cascade cpWER per resample",
            "acceleration": "delete-1 jackknife (in-sample ensemble cascade cpWER on n-1)",
            "bias_correction": "z0 = Phi^{-1}( #{boot < theta_hat} / B ), clamped to (0.5/B, 1-0.5/B)",
            "normal_inverse": "Acklam rational approximation + 1 Halley step (no scipy)",
        },
        "rq43_original_rule_reference": {
            "kl_threshold": RQ43_KL_THRESHOLD,
            "cascade_cpwer": RQ43_CASCADE_CPWER,
            "baseline_cpwer": RQ43_BASELINE_CPWER,
            "base_ratio": RQ43_BASE_RATIO,
            "reproduced_in_sample": round(rq43_cas, 6),
            "rule": "max sensitivity at >= 90% specificity (RQ43's n=3 kl_sep)",
        },
        "rq46_original_rule_ci_reference": {
            **rq46_ref,
            "method": "percentile CI (RQ46), in-bag at fixed threshold 3.30",
            "note": ("RQ46's anchor is a percentile CI evaluated in-bag at the FIXED "
                     "threshold 3.30. RQ62's BCa CI is bias-corrected + accelerated "
                     "and evaluated OOB at the RE-CALIBRATED ensemble thresholds. "
                     "The H62c comparison is therefore directional, not a pure "
                     "CI-method swap (same caveat as RQ54's H54b and RQ59's H59c)."),
        },
        "rq54_f1_reference": {
            "detector": "kl_sep (RQ43's n=3 KL)",
            **rq54_ref,
            "note": ("RQ54's F1 cascade on the KL detector: 83.1% escalation, "
                     "OOB median cpWER 0.780, BCa width 0.2481. RQ62 asks whether "
                     "the ensemble is less aggressive (H62a) while maintaining "
                     "robustness (H62b/c)."),
        },
        "rq59_youdens_j_reference": {
            "detector": "kl_sep (RQ43's n=3 KL)",
            **rq59_ref,
            "note": ("RQ59's Youden's J cascade on the KL detector: 83.1% "
                     "escalation (flat-topped ROC collapse), OOB median cpWER "
                     "~0.782. RQ62 tests whether the ensemble escapes the "
                     "flat-topped-ROC collapse by using a second detector."),
        },
        "in_sample_calibration": {
            "kl": {
                "threshold": round(kl_thr_in, 6),
                "specificity": round(float(kl_cal["specificity"]), 6),
                "n_neg": int(kl_cal["n_neg"]),
                "max_fp": int(kl_cal["max_fp"]),
                "calibration_rule": "RQ58 candidate-set >=90% specificity",
            },
            "lang_id": {
                "threshold": round(lang_thr_in, 6),
                "sensitivity": round(float(lang_cal["sensitivity"]), 6),
                "specificity": round(float(lang_cal["specificity"]), 6),
                "tp": int(lang_cal["tp"]), "fp": int(lang_cal["fp"]),
                "tn": int(lang_cal["tn"]), "fn": int(lang_cal["fn"]),
                "calibration_rule": "RQ44 grid >=90% specificity, max sensitivity",
            },
        },
        "in_sample_single_detector_escalation": {
            "kl_only": {"count": n_esc_kl_only, "fraction": round(frac_kl_only, 6)},
            "lang_only": {"count": n_esc_lang_only, "fraction": round(frac_lang_only, 6)},
        },
        "in_sample_ensemble": {
            "or": {
                "kl_threshold": round(kl_thr_in, 6),
                "lang_threshold": round(lang_thr_in, 6),
                "n_escalated": n_esc_or,
                "escalation_fraction": round(frac_or, 6),
                "cascade_cpwer": round(theta_hat_or, 6),
                "cascade_compute": round(compute_or, 6),
            },
            "and": {
                "kl_threshold": round(kl_thr_in, 6),
                "lang_threshold": round(lang_thr_in, 6),
                "n_escalated": n_esc_and,
                "escalation_fraction": round(frac_and, 6),
                "cascade_cpwer": round(theta_hat_and, 6),
                "cascade_compute": round(compute_and, 6),
            },
        },
        "bootstrap_threshold_distributions": {
            "kl": {
                **_finite_stats(boot_kl_thr),
                "n_unique": int(np.unique(boot_kl_thr).size),
                "n_modes_5pct": kl_modes["n_modes"],
                "modes_5pct": kl_modes["modes"],
                "min_mode_fraction": float(MIN_MODE_FRACTION),
                "note": ("+inf threshold = no finite KL candidate met the >=90% "
                         "specificity floor on that resample's in-bag set (no KL "
                         "escalation). inf is a legitimate mode and is counted by "
                         "count_modes; descriptive stats are on the finite subset."),
            },
            "lang_id": {
                **_finite_stats(boot_lang_thr),
                "n_unique": int(np.unique(boot_lang_thr).size),
                "n_modes_5pct": lang_modes["n_modes"],
                "modes_5pct": lang_modes["modes"],
                "min_mode_fraction": float(MIN_MODE_FRACTION),
            },
        },
        "bootstrap_oob_cpwer_distributions": {
            "or": {
                "n_valid": int(np.sum(~np.isnan(boot_oob_or))),
                "median": round(oob_median_or, 6),
                "mean": round(oob_mean_or, 6),
                "min": round(float(np.nanmin(boot_oob_or)), 6),
                "max": round(float(np.nanmax(boot_oob_or)), 6),
                "p2_5": round(float(np.nanpercentile(boot_oob_or, 2.5)), 6),
                "p97_5": round(float(np.nanpercentile(boot_oob_or, 97.5)), 6),
            },
            "and": {
                "n_valid": int(np.sum(~np.isnan(boot_oob_and))),
                "median": round(oob_median_and, 6),
                "mean": round(oob_mean_and, 6),
                "min": round(float(np.nanmin(boot_oob_and)), 6),
                "max": round(float(np.nanmax(boot_oob_and)), 6),
                "p2_5": round(float(np.nanpercentile(boot_oob_and, 2.5)), 6),
                "p97_5": round(float(np.nanpercentile(boot_oob_and, 97.5)), 6),
            },
        },
        "bca_ci": {
            "or": {
                "lo": round(bca_or["lo"], 6),
                "hi": round(bca_or["hi"], 6),
                "width": round(bca_width_or, 6),
                "median": round(bca_or["median"], 6),
                "z0": round(bca_or["z0"], 6) if np.isfinite(bca_or["z0"]) else None,
                "accel": round(bca_or["accel"], 6),
                "alpha1": round(bca_or["alpha1"], 6) if np.isfinite(bca_or["alpha1"]) else None,
                "alpha2": round(bca_or["alpha2"], 6) if np.isfinite(bca_or["alpha2"]) else None,
                "method": bca_or["method"],
                "theta_hat": round(theta_hat_or, 6),
                "n_valid": int(bca_or["n_valid"]),
            },
            "and": {
                "lo": round(bca_and["lo"], 6),
                "hi": round(bca_and["hi"], 6),
                "width": round(bca_width_and, 6),
                "median": round(bca_and["median"], 6),
                "z0": round(bca_and["z0"], 6) if np.isfinite(bca_and["z0"]) else None,
                "accel": round(bca_and["accel"], 6),
                "alpha1": round(bca_and["alpha1"], 6) if np.isfinite(bca_and["alpha1"]) else None,
                "alpha2": round(bca_and["alpha2"], 6) if np.isfinite(bca_and["alpha2"]) else None,
                "method": bca_and["method"],
                "theta_hat": round(theta_hat_and, 6),
                "n_valid": int(bca_and["n_valid"]),
            },
        },
        "jackknife": {
            "or": {
                "accel": round(accel_or, 6),
                "theta_loo_mean": round(float(np.mean(theta_loo_or)), 6),
                "theta_loo_min": round(float(np.min(theta_loo_or)), 6),
                "theta_loo_max": round(float(np.max(theta_loo_or)), 6),
            },
            "and": {
                "accel": round(accel_and, 6),
                "theta_loo_mean": round(float(np.mean(theta_loo_and)), 6),
                "theta_loo_min": round(float(np.min(theta_loo_and)), 6),
                "theta_loo_max": round(float(np.max(theta_loo_and)), 6),
            },
        },
        "hypothesis_verdicts": {
            "H62a": {
                "statement": ("Ensemble cascade escalates < 83.1% of windows to base "
                              "(less aggressive than KL-alone RQ59's 83.1%)"),
                "primary_gate": "or",
                "or": {
                    "escalation_fraction": round(frac_or, 6),
                    "max_escalation": H62A_MAX_ESCALATION,
                    "supported": bool(h62a_supported_or),
                },
                "and": {
                    "escalation_fraction": round(frac_and, 6),
                    "max_escalation": H62A_MAX_ESCALATION,
                    "supported": bool(h62a_supported_and),
                },
                "rq59_reference_escalation": RQ59_YOUDENS_J_ESCALATION,
                "rq54_reference_escalation": RQ54_F1_ESCALATION,
                "kill": f"escalation fraction >= {H62A_MAX_ESCALATION}",
                "supported": bool(h62a_supported_or),
            },
            "H62b": {
                "statement": ("Ensemble cascade OOB cpWER <= 0.889 (matches RQ43's "
                              "original-rule cpWER 0.888947)"),
                "primary_gate": "or",
                "or": {
                    "median_cpwer": round(oob_median_or, 6),
                    "max_cpwer": H62B_MAX_CPWER,
                    "supported": bool(h62b_supported_or),
                },
                "and": {
                    "median_cpwer": round(oob_median_and, 6),
                    "max_cpwer": H62B_MAX_CPWER,
                    "supported": bool(h62b_supported_and),
                },
                "rq43_reference_cpwer": RQ43_CASCADE_CPWER,
                "kill": f"median cpWER > {H62B_MAX_CPWER}",
                "supported": bool(h62b_supported_or),
            },
            "H62c": {
                "statement": ("Ensemble cascade BCa CI width <= 0.2489 (maintains "
                              "robustness vs RQ46's original-rule width)"),
                "primary_gate": "or",
                "or": {
                    "bca_ci_width": round(bca_width_or, 6),
                    "max_width": H62C_MAX_WIDTH,
                    "supported": bool(h62c_supported_or),
                },
                "and": {
                    "bca_ci_width": round(bca_width_and, 6),
                    "max_width": H62C_MAX_WIDTH,
                    "supported": bool(h62c_supported_and),
                },
                "rq46_reference_width": RQ46_CI_WIDTH,
                "kill": f"BCa width > {H62C_MAX_WIDTH}",
                "supported": bool(h62c_supported_or),
            },
        },
        "per_bootstrap": {
            "kl_thresholds": [round(float(t), 6) for t in boot_kl_thr],
            "lang_thresholds": [round(float(t), 6) for t in boot_lang_thr],
            "oob_cpwer_or": [round(float(c), 6) if not math.isnan(float(c)) else None
                             for c in boot_oob_or],
            "oob_cpwer_and": [round(float(c), 6) if not math.isnan(float(c)) else None
                              for c in boot_oob_and],
            "n_oob": [int(x) for x in boot["n_oob"]],
        },
    }
    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_bootstrap_csv(
        OUT_CSV, boot["boot_idx"], boot_kl_thr, boot_lang_thr,
        boot_oob_or, boot_oob_and, boot["n_oob"],
        boot["n_escalated_oob_or"], boot["n_escalated_oob_and"], n)

    # --- console
    print(f"=== RQ62: Cascade with KL+lang-id ensemble gate ===")
    print(f"Label: experimental/frontier  |  n={n} AISHELL-4 windows "
          f"({n_hall} hall / {n_clean} clean)")
    print(f"Controlled comparison: only the gate changes vs RQ59 (ensemble vs single-KL Youden's J).")
    print(f"RQ43 original rule @ kl_sep>=3.30: cpwer={rq43_cas:.4f} "
          f"(reproduces {RQ43_CASCADE_CPWER})")
    print()
    print(f"In-sample calibration (full 77 windows):")
    print(f"  KL threshold (RQ58 rule)     = {kl_thr_in:.6f}  "
          f"(spec {kl_cal['specificity']:.4f}, max_fp {kl_cal['max_fp']})")
    print(f"  lang-id threshold (RQ44 rule)= {lang_thr_in:.6f}  "
          f"(sens {lang_cal['sensitivity']:.4f}, spec {lang_cal['specificity']:.4f})")
    print()
    print(f"In-sample single-detector escalation:")
    print(f"  KL alone (>= {kl_thr_in:.4f}): {n_esc_kl_only}/77 = {frac_kl_only:.4f}")
    print(f"  lang alone (>= {lang_thr_in:.4f}): {n_esc_lang_only}/77 = {frac_lang_only:.4f}")
    print()
    print(f"In-sample ensemble cascade:")
    print(f"  OR  gate: {n_esc_or}/77 = {frac_or:.4f} escalated  "
          f"cpWER={theta_hat_or:.4f}  compute={compute_or:.4f}x")
    print(f"  AND gate: {n_esc_and}/77 = {frac_and:.4f} escalated  "
          f"cpWER={theta_hat_and:.4f}  compute={compute_and:.4f}x")
    print()
    kl_fs = _finite_stats(boot_kl_thr)
    lang_fs = _finite_stats(boot_lang_thr)
    print(f"Bootstrap B={N_BOOT} seed={SEED} (OOB, re-calibrated thresholds per resample):")
    print(f"  KL threshold: median(finite)={kl_fs['median']:.4f}  "
          f"n_unique={np.unique(boot_kl_thr).size}  n_modes>=5%={kl_modes['n_modes']}  "
          f"inf_frac={kl_fs['inf_fraction']:.3f}")
    for m in kl_modes["modes"]:
        t_str = "inf" if math.isinf(m["threshold"]) else f"{m['threshold']:.4f}"
        print(f"    mode KL={t_str}  count={m['count']}  frac={m['fraction']:.3f}")
    print(f"  lang-id threshold: median={lang_fs['median']:.4f}  "
          f"n_unique={np.unique(boot_lang_thr).size}  n_modes>=5%={lang_modes['n_modes']}")
    for m in lang_modes["modes"]:
        t_str = "inf" if math.isinf(m["threshold"]) else f"{m['threshold']:.4f}"
        print(f"    mode lang={t_str}  count={m['count']}  frac={m['fraction']:.3f}")
    print(f"  OR  OOB cpWER: median={oob_median_or:.4f}  mean={oob_mean_or:.4f}  "
          f"pct[{np.nanpercentile(boot_oob_or,2.5):.4f},{np.nanpercentile(boot_oob_or,97.5):.4f}]")
    print(f"  AND OOB cpWER: median={oob_median_and:.4f}  mean={oob_mean_and:.4f}  "
          f"pct[{np.nanpercentile(boot_oob_and,2.5):.4f},{np.nanpercentile(boot_oob_and,97.5):.4f}]")
    print(f"  OR  BCa CI: [{bca_or['lo']:.4f}, {bca_or['hi']:.4f}]  width={bca_width_or:.4f}  "
          f"(z0={bca_or['z0']:.4f}, a={accel_or:.4f}, method={bca_or['method']})")
    print(f"  AND BCa CI: [{bca_and['lo']:.4f}, {bca_and['hi']:.4f}]  width={bca_width_and:.4f}  "
          f"(z0={bca_and['z0']:.4f}, a={accel_and:.4f}, method={bca_and['method']})")
    print()
    print("Hypothesis verdicts (primary gate = OR):")
    print(f"  H62a (escalation < {H62A_MAX_ESCALATION:.1%}):  "
          f"OR {'SUPPORTED' if h62a_supported_or else 'KILLED'} "
          f"(frac={frac_or:.4f})  |  AND {'SUPPORTED' if h62a_supported_and else 'KILLED'} "
          f"(frac={frac_and:.4f})  [RQ59 ref={RQ59_YOUDENS_J_ESCALATION:.4f}]")
    print(f"  H62b (OOB cpWER <= {H62B_MAX_CPWER}):  "
          f"OR {'SUPPORTED' if h62b_supported_or else 'KILLED'} "
          f"(median={oob_median_or:.4f})  |  AND {'SUPPORTED' if h62b_supported_and else 'KILLED'} "
          f"(median={oob_median_and:.4f})  [RQ43 ref={RQ43_CASCADE_CPWER}]")
    print(f"  H62c (BCa width <= {H62C_MAX_WIDTH}):  "
          f"OR {'SUPPORTED' if h62c_supported_or else 'KILLED'} "
          f"(width={bca_width_or:.4f})  |  AND {'SUPPORTED' if h62c_supported_and else 'KILLED'} "
          f"(width={bca_width_and:.4f})  [RQ46 ref={RQ46_CI_WIDTH}]")
    print()
    print(f"RQ59 Youden's J reference: thr={rq59_ref.get('youdens_j_threshold')}  "
          f"frac={rq59_ref['youdens_j_escalation']:.4f}  "
          f"cpwer={rq59_ref['youdens_j_oob_median_cpwer']:.4f}  "
          f"width={rq59_ref.get('youdens_j_bca_width', '?')}")
    print(f"RQ54 F1 reference: thr={rq54_ref.get('f1_threshold')}  "
          f"frac={rq54_ref['f1_escalation']:.4f}  "
          f"cpwer={rq54_ref['f1_oob_median_cpwer']:.4f}  "
          f"width={rq54_ref['f1_bca_width']:.4f}")
    print(f"RQ46 original-rule reference: CI=[{rq46_ref['ci_lo']:.4f},{rq46_ref['ci_hi']:.4f}] "
          f"width={rq46_ref['width']:.4f} (percentile, in-bag, fixed thr=3.30)")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
