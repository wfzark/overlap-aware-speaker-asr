"""RQ38: Speaker-count effect on hallucination rate (AISHELL-4).

REANALYSIS ONLY — no Whisper / no ASR model is run. This script reads the existing
AISHELL-4 external-validation results
(``results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json``,
label ``external/sanity-check``, PR #890) and asks whether the number of active
speakers per window correlates with the separated-track hallucination rate, and
whether speaker count predicts Mode S specifically.

Label: experimental/frontier. Closes #945.

Research question
-----------------
Does the number of active speakers per window correlate with hallucination rate on
AISHELL-4, and does speaker count predict Mode S specifically?

Pre-registered hypotheses
-------------------------
- H38a: Positive correlation between active speaker count and hallucination rate
  (Spearman rho > 0.2). More speakers -> more silence gaps -> more hallucination.
- H38b: Mode S windows (w22, w30) have <= 2 active speakers (Mode S is a
  low-speaker-count phenomenon, consistent with RQ37's finding that speaker 001-M
  dominates).
- H38c: The speaker-count effect is mediated by silence-gap fraction, not by
  multi-speaker confusion (partial correlation of speaker count with hallucination,
  controlling for silence fraction, drops below 0.1).

Background
----------
RQ12 (``results/frontier/router_failure_modes/FINDINGS.md``) found hallucination is
driven by silence gaps, not multi-speaker confusion: only 2 of 11 router failure
windows had > 2 active speakers, and 3 failure windows had just 1 speaker. RQ19
(``results/frontier/mode_s_detector/FINDINGS.md``) identified the 2 Mode S residual
windows (22, 30) as low-lang-id-entropy monoscript-Chinese near-duplicates of the
mixed decode. RQ37 (per-speaker cpWER decomposition, referenced in the task brief)
reports that speaker 001-M dominates the AISHELL-4 reference. RQ38 directly tests
the speaker-count correlation that these prior studies touch on only indirectly.

Method (pure reanalysis, numpy + scipy + stdlib only)
-----------------------------------------------------
For each of the 77 windows:
- ``active_speakers``  = count of non-empty ``separated_text_per_speaker`` values.
- ``num_speakers``     = speakers present per the oracle TextGrid (the JSON field).
- ``silence_fraction`` = (num_speakers - active_speakers) / num_speakers  (fraction
  of present speakers for whom Whisper produced an empty separated transcript).
- ``length_ratio``     = separated_total_length / mixed_text_length  (a second
  silence-gap proxy, more independent of speaker count than the per-speaker
  fraction).
- ``hallucinated``     = ``always_separated_cpwer > 1.0`` (RQ12's split: 37/77).
- ``mode_s``           = hallucinated AND window_id in {22, 30} (RQ19's definition).

We then compute:
- Spearman rho between ``active_speakers`` and (a) ``always_separated_cpwer`` and
  (b) the binary ``hallucinated`` label, each with a 10,000-permutation test
  (seed=42). The same is reported for ``num_speakers`` as a secondary measure.
- Hallucination rate per active-speaker stratum {0, 1, 2, 3+} with bootstrap 95%
  CIs (10,000 resamples, seed=42).
- Mode S windows' active-speaker count (H38b).
- Rank-based partial correlation of ``active_speakers`` with ``hallucinated``,
  controlling for ``silence_fraction`` (primary) and ``length_ratio`` (robustness),
  plus a subsample analysis restricted to ``active_speakers >= 1`` (which removes
  the structural floor where all-empty decodes score cpWER = 1.0 exactly).

Structural confound (disclosed honestly)
----------------------------------------
When Whisper produces no output for any speaker (``active_speakers == 0``) the
separated cpWER is exactly 1.0 (pure deletions, no insertions), so the window is
mechanically non-hallucinated by the ``> 1.0`` label. 13 of 77 windows sit at this
floor. This inflates the positive speaker-count / hallucination correlation and is
the load-bearing caveat for H38a and H38c. The ``active >= 1`` subsample and the
length-ratio robustness check both address it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_JSON = (
    PROJECT_ROOT
    / "results"
    / "external_sanity_check"
    / "aishell4"
    / "rq1_aishell4_validation_results.json"
)
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "speaker_count_effect_results.csv"
OUT_JSON = OUT_DIR / "speaker_count_effect_results.json"

# --------------------------------------------------------------------------- constants
HALLUCINATION_CPWER_THRESHOLD = 1.0
MODE_S_WINDOW_IDS: tuple[int, ...] = (22, 30)
N_BOOT = 10_000
N_PERM = 10_000
SEED = 42
BOOTSTRAP_CI = 0.95
# Active-speaker strata: (lo, hi, label). 3+ collapses 3,4,6 into one bucket.
ACTIVE_STRATA: tuple[tuple[int, int, str], ...] = (
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 10**9, "3+"),
)


# ===========================================================================
# Pure helpers (no IO; tested in tests/test_speaker_count_effect.py)
# ===========================================================================
def count_active_speakers(separated_text_per_speaker: dict[str, str]) -> int:
    """Count speakers with a non-empty separated transcript.

    "Non-empty" means the stripped string has positive length. Whitespace-only
    transcripts count as empty (Whisper produced no real content).
    """
    if not separated_text_per_speaker:
        return 0
    return sum(1 for v in separated_text_per_speaker.values() if v and str(v).strip())


def compute_silence_fraction(num_speakers: int, active_speakers: int) -> float:
    """Fraction of present speakers with an empty separated transcript.

    Returns 0.0 when ``num_speakers`` is 0 (no speakers present -> no silence
    attributable to a missing speaker decode). ``active_speakers`` is clamped to
    ``[0, num_speakers]`` for safety.
    """
    if num_speakers <= 0:
        return 0.0
    active = max(0, min(int(active_speakers), int(num_speakers)))
    return (num_speakers - active) / float(num_speakers)


def compute_length_ratio(separated_total_length: int, mixed_text_length: int) -> float:
    """Ratio of separated-decode length to mixed-decode length.

    A second silence-gap proxy: low values mean Whisper under-produced on the
    separated track (silence-induced short decodes); high values mean it
    over-produced (insertion-heavy hallucination). Guarded against zero
    denominators (returns 0.0 when mixed length is 0).
    """
    if mixed_text_length <= 0:
        return 0.0
    return float(separated_total_length) / float(mixed_text_length)


def is_hallucinated(
    always_separated_cpwer: float,
    threshold: float = HALLUCINATION_CPWER_THRESHOLD,
) -> bool:
    """Hallucination label: separated cpWER strictly above ``threshold``.

    Uses strict ``>`` so that the all-empty-decode cpWER of exactly 1.0 (pure
    deletions) is labelled NON-hallucinated, matching RQ12's 37/77 split.
    """
    return float(always_separated_cpwer) > float(threshold)


def is_mode_s_window(
    window_id: int,
    hallucinated: bool,
    mode_s_ids: Sequence[int] = MODE_S_WINDOW_IDS,
) -> bool:
    """Mode S membership: hallucinated AND window_id in the Mode S set."""
    return bool(hallucinated) and int(window_id) in set(int(s) for s in mode_s_ids)


def extract_window_features(window: dict[str, Any]) -> dict[str, Any]:
    """Extract all per-window features used by the analysis from one JSON window.

    Pure: given a window dict, returns a feature dict. Does no IO.
    """
    sep = window.get("separated_text_per_speaker", {}) or {}
    num_speakers = int(window.get("num_speakers", len(sep)))
    active = count_active_speakers(sep)
    cpwer = float(window.get("always_separated_cpwer", 0.0))
    halluc = is_hallucinated(cpwer)
    wid = int(window.get("window_id", -1))
    return {
        "window_id": wid,
        "num_speakers": num_speakers,
        "active_speakers": active,
        "empty_speakers": num_speakers - active,
        "silence_fraction": compute_silence_fraction(num_speakers, active),
        "length_ratio": compute_length_ratio(
            int(window.get("separated_total_length", 0)),
            int(window.get("mixed_text_length", 0)),
        ),
        "runtime_ratio": float(window.get("runtime_ratio", 0.0)),
        "always_separated_cpwer": cpwer,
        "always_mixed_cpwer": float(window.get("always_mixed_cpwer", 0.0)),
        "hallucinated": halluc,
        "mode_s": is_mode_s_window(wid, halluc),
        "overlap_label": window.get("overlap_label", ""),
    }


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman correlation (rho, two-sided p-value).

    Returns (0.0, 1.0) for degenerate inputs (length < 2 or zero variance) so the
    pipeline never crashes on a constant column.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2 or ya.size < 2:
        return 0.0, 1.0
    if np.nanstd(xa) == 0.0 or np.nanstd(ya) == 0.0:
        return 0.0, 1.0
    res = spearmanr(xa, ya)
    rho = float(res.statistic if hasattr(res, "statistic") else res[0])
    p = float(res.pvalue if hasattr(res, "pvalue") else res[1])
    if not np.isfinite(rho):
        return 0.0, 1.0
    return rho, p


def _rank(a: Sequence[float]) -> np.ndarray:
    """Average-rank transform with NaN handled last (tied at the bottom)."""
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    ranks = np.full(arr.shape, np.nan, dtype=float)
    if finite_mask.any():
        ranks[finite_mask] = rankdata(arr[finite_mask], method="average")
    return ranks


def partial_correlation(
    x: Sequence[float], y: Sequence[float], z: Sequence[float]
) -> float:
    """Rank-based partial correlation of x and y, controlling for z.

    All three variables are rank-transformed, then z is partialled out of both x
    and y by ordinary least-squares residualisation, and the Pearson correlation
    of the two residual vectors is returned. This is the standard nonparametric
    partial correlation. Returns 0.0 for degenerate inputs.
    """
    rx = _rank(x)
    ry = _rank(y)
    rz = _rank(z)
    if rx.size < 3 or ry.size < 3 or rz.size < 3:
        return 0.0
    mask = np.isfinite(rx) & np.isfinite(ry) & np.isfinite(rz)
    if mask.sum() < 3:
        return 0.0
    rx, ry, rz = rx[mask], ry[mask], rz[mask]
    if np.std(rx) == 0.0 or np.std(ry) == 0.0 or np.std(rz) == 0.0:
        # If z is constant, residualising is a no-op; fall back to the raw rank
        # correlation of x and y. If x or y is constant, correlation is 0.
        if np.std(rz) == 0.0:
            if np.std(rx) == 0.0 or np.std(ry) == 0.0:
                return 0.0
            return float(pearsonr(rx, ry)[0])
        return 0.0

    def _residualise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Linear regression a ~ 1 + b; return residuals.
        design = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rx_res = _residualise(rx, rz)
    ry_res = _residualise(ry, rz)
    # Guard against (near-)perfect collinearity: if residualising leaves no
    # meaningful variance, the partial correlation is undefined -> return 0.0.
    # The 1e-10 tolerance handles floating-point collinearity (e.g. z == x).
    if np.std(rx_res) < 1e-10 or np.std(ry_res) < 1e-10:
        return 0.0
    return float(pearsonr(rx_res, ry_res)[0])


def permutation_test_spearman(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> tuple[float, float]:
    """Spearman rho with a permutation p-value (two-sided).

    Permutes ``y`` and recomputes Spearman rho; p-value = fraction of permutations
    with |rho_perm| >= |rho_obs|, with +1 smoothing. Returns (rho_obs, p_value).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = xa.size
    if n < 2 or np.nanstd(xa) == 0.0 or np.nanstd(ya) == 0.0:
        return 0.0, 1.0
    rho_obs, _ = spearman_rho(xa, ya)
    rng = np.random.default_rng(seed)
    count_extreme = 0
    abs_obs = abs(rho_obs)
    for _ in range(int(n_perm)):
        yp = rng.permutation(ya)
        if np.std(yp) == 0.0:
            continue
        rho_p, _ = spearman_rho(xa, yp)
        if abs(rho_p) >= abs_obs:
            count_extreme += 1
    p_value = (count_extreme + 1.0) / (n_perm + 1.0)
    return rho_obs, float(p_value)


def bootstrap_rate_ci(
    labels: Sequence[int] | Sequence[bool] | np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of a 0/1 label vector.

    Returns (point_estimate, lower, upper) at the given central confidence level.
    Returns (0.0, 0.0, 0.0) for empty input.
    """
    arr = np.asarray(labels, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return point, lower, upper


def stratify_by_count(
    active_counts: Sequence[int],
    halluc_labels: Sequence[int] | Sequence[bool],
    strata: Sequence[tuple[int, int, str]] = ACTIVE_STRATA,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    ci: float = BOOTSTRAP_CI,
) -> list[dict[str, Any]]:
    """Hallucination rate per active-speaker stratum with bootstrap 95% CI.

    Each stratum is (lo, hi, label); a window with ``lo <= active <= hi`` falls in
    that stratum. Returns one dict per stratum: ``{stratum, n, halluc, rate,
    ci_low, ci_high}``. Empty strata are returned with ``n = 0`` and NaN rate.
    """
    active = np.asarray(active_counts, dtype=int)
    labels = np.asarray(halluc_labels, dtype=int)
    rows: list[dict[str, Any]] = []
    for lo, hi, label in strata:
        mask = (active >= lo) & (active <= hi)
        sub = labels[mask]
        if sub.size == 0:
            rows.append(
                {
                    "stratum": label,
                    "n": 0,
                    "halluc": 0,
                    "rate": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                }
            )
            continue
        rate, lo_ci, hi_ci = bootstrap_rate_ci(sub, n_boot=n_boot, seed=seed, ci=ci)
        rows.append(
            {
                "stratum": label,
                "n": int(sub.size),
                "halluc": int(sub.sum()),
                "rate": float(rate),
                "ci_low": float(lo_ci),
                "ci_high": float(hi_ci),
            }
        )
    return rows


def evaluate_hypotheses(
    rho_active_cpwer: float,
    perm_p_active_cpwer: float,
    rho_active_halluc: float,
    perm_p_active_halluc: float,
    rho_num_halluc: float,
    mode_s_active_counts: dict[int, int],
    mode_s_num_counts: dict[int, int],
    partial_active_silence: float,
    partial_active_lengthratio: float,
    partial_active_silence_active_ge1: float,
) -> dict[str, Any]:
    """Evaluate H38a / H38b / H38c from the computed statistics.

    Pure decision logic; takes already-computed numbers so it can be unit-tested
    independently of the data.
    """
    h38a_supported = rho_active_halluc > 0.2 and perm_p_active_halluc < 0.05
    h38a_cpwer_supported = rho_active_cpwer > 0.2 and perm_p_active_cpwer < 0.05

    mode_s_all_low = all(
        c <= 2 for c in list(mode_s_active_counts.values()) + list(mode_s_num_counts.values())
    )
    h38b_supported = bool(mode_s_all_low) and len(mode_s_active_counts) > 0

    h38c_supported = abs(partial_active_silence) < 0.1
    h38c_lengthratio_supported = abs(partial_active_lengthratio) < 0.1

    return {
        "H38a": {
            "claim": "Positive correlation between active speaker count and hallucination rate (Spearman rho > 0.2).",
            "rho_active_vs_hallucination": rho_active_halluc,
            "perm_p_active_vs_hallucination": perm_p_active_halluc,
            "rho_active_vs_cpwer": rho_active_cpwer,
            "perm_p_active_vs_cpwer": perm_p_active_cpwer,
            "rho_num_vs_hallucination": rho_num_halluc,
            "threshold_rho": 0.2,
            "supported": bool(h38a_supported),
            "supported_on_cpwer": bool(h38a_cpwer_supported),
            "verdict": "SUPPORTED" if h38a_supported else "NOT SUPPORTED",
        },
        "H38b": {
            "claim": "Mode S windows (w22, w30) have <= 2 active speakers.",
            "mode_s_active_speaker_counts": mode_s_active_counts,
            "mode_s_num_speaker_counts": mode_s_num_counts,
            "supported": bool(h38b_supported),
            "verdict": "SUPPORTED" if h38b_supported else "NOT SUPPORTED",
        },
        "H38c": {
            "claim": "Speaker-count effect mediated by silence fraction (partial rho < 0.1).",
            "partial_active_vs_halluc_controlling_silence": partial_active_silence,
            "partial_active_vs_halluc_controlling_lengthratio": partial_active_lengthratio,
            "partial_active_vs_halluc_controlling_silence_active_ge1": partial_active_silence_active_ge1,
            "threshold_abs_partial": 0.1,
            "supported_primary": bool(h38c_supported),
            "supported_lengthratio": bool(h38c_lengthratio_supported),
            "verdict": (
                "SUPPORTED"
                if h38c_supported
                else "NOT SUPPORTED"
            ),
        },
    }


# ===========================================================================
# IO / main
# ===========================================================================
def load_windows(path: Path = SRC_JSON) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["windows"]


def run_analysis() -> dict[str, Any]:
    """Run the full RQ38 analysis and write CSV + JSON outputs."""
    windows = load_windows()
    features = [extract_window_features(w) for w in windows]
    n = len(features)

    active = np.array([f["active_speakers"] for f in features], dtype=float)
    num_spk = np.array([f["num_speakers"] for f in features], dtype=float)
    cpwer = np.array([f["always_separated_cpwer"] for f in features], dtype=float)
    halluc = np.array([1 if f["hallucinated"] else 0 for f in features], dtype=float)
    silence = np.array([f["silence_fraction"] for f in features], dtype=float)
    lenratio = np.array([f["length_ratio"] for f in features], dtype=float)

    # --- H38a: Spearman correlations + permutation tests -------------------
    rho_active_cpwer, p_active_cpwer = permutation_test_spearman(active, cpwer)
    rho_active_halluc, p_active_halluc = permutation_test_spearman(active, halluc)
    # Secondary measure: num_speakers (configured speaker count) vs hallucination.
    rho_num_halluc, p_num_halluc = permutation_test_spearman(num_spk, halluc)
    rho_num_cpwer, p_num_cpwer = permutation_test_spearman(num_spk, cpwer)
    # Silence-fraction vs hallucination (for the mediation argument).
    rho_silence_halluc, p_silence_halluc = permutation_test_spearman(silence, halluc)
    rho_lenratio_halluc, p_lenratio_halluc = permutation_test_spearman(lenratio, halluc)

    # --- Stratification -----------------------------------------------------
    strata_active = stratify_by_count(active.astype(int), halluc.astype(int))
    strata_num = stratify_by_count(
        num_spk.astype(int),
        halluc.astype(int),
        strata=((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 10**9, "5+")),
    )

    # --- H38b: Mode S speaker counts ---------------------------------------
    mode_s_active: dict[int, int] = {}
    mode_s_num: dict[int, int] = {}
    for f in features:
        if f["mode_s"]:
            mode_s_active[f["window_id"]] = int(f["active_speakers"])
            mode_s_num[f["window_id"]] = int(f["num_speakers"])

    # --- H38c: partial correlations ----------------------------------------
    partial_silence = partial_correlation(active, halluc, silence)
    partial_lenratio = partial_correlation(active, halluc, lenratio)
    # Subsample: active >= 1 (removes the structural cpWER=1.0 floor).
    ge1_mask = active >= 1
    partial_silence_ge1 = partial_correlation(
        active[ge1_mask], halluc[ge1_mask], silence[ge1_mask]
    )
    partial_lenratio_ge1 = partial_correlation(
        active[ge1_mask], halluc[ge1_mask], lenratio[ge1_mask]
    )
    # Also report the simple (non-partial) correlation in the active>=1 subsample,
    # to show how much of H38a's rho survives removing the all-empty floor.
    rho_active_halluc_ge1, p_active_halluc_ge1 = permutation_test_spearman(
        active[ge1_mask], halluc[ge1_mask]
    )

    # --- Hypothesis evaluation ---------------------------------------------
    verdicts = evaluate_hypotheses(
        rho_active_cpwer=rho_active_cpwer,
        perm_p_active_cpwer=p_active_cpwer,
        rho_active_halluc=rho_active_halluc,
        perm_p_active_halluc=p_active_halluc,
        rho_num_halluc=rho_num_halluc,
        mode_s_active_counts=mode_s_active,
        mode_s_num_counts=mode_s_num,
        partial_active_silence=partial_silence,
        partial_active_lengthratio=partial_lenratio,
        partial_active_silence_active_ge1=partial_silence_ge1,
    )

    summary = {
        "label": "experimental/frontier",
        "rq": "RQ38",
        "closes_issue": 945,
        "dataset": "AISHELL-4",
        "meeting_id": "M_R003S02C01",
        "n_windows": n,
        "n_hallucinated": int(halluc.sum()),
        "hallucination_threshold": HALLUCINATION_CPWER_THRESHOLD,
        "active_speaker_distribution": {
            str(k): int(v)
            for k, v in zip(*np.unique(active.astype(int), return_counts=True))
        },
        "num_speakers_distribution": {
            str(k): int(v)
            for k, v in zip(*np.unique(num_spk.astype(int), return_counts=True))
        },
        "mode_s_window_ids": list(MODE_S_WINDOW_IDS),
        "hypotheses": verdicts,
        "correlations": {
            "active_vs_cpwer": {
                "spearman_rho": rho_active_cpwer,
                "perm_p": p_active_cpwer,
                "n_perm": N_PERM,
                "seed": SEED,
            },
            "active_vs_hallucination": {
                "spearman_rho": rho_active_halluc,
                "perm_p": p_active_halluc,
                "n_perm": N_PERM,
                "seed": SEED,
            },
            "num_speakers_vs_hallucination": {
                "spearman_rho": rho_num_halluc,
                "perm_p": p_num_halluc,
            },
            "num_speakers_vs_cpwer": {
                "spearman_rho": rho_num_cpwer,
                "perm_p": p_num_cpwer,
            },
            "silence_fraction_vs_hallucination": {
                "spearman_rho": rho_silence_halluc,
                "perm_p": p_silence_halluc,
            },
            "length_ratio_vs_hallucination": {
                "spearman_rho": rho_lenratio_halluc,
                "perm_p": p_lenratio_halluc,
            },
        },
        "stratification_active_speakers": strata_active,
        "stratification_num_speakers": strata_num,
        "partial_correlations": {
            "active_vs_halluc_controlling_silence": partial_silence,
            "active_vs_halluc_controlling_lengthratio": partial_lenratio,
            "active_vs_halluc_controlling_silence_active_ge1": partial_silence_ge1,
            "active_vs_halluc_controlling_lengthratio_active_ge1": partial_lenratio_ge1,
            "active_vs_halluc_simple_active_ge1": {
                "spearman_rho": rho_active_halluc_ge1,
                "perm_p": p_active_halluc_ge1,
            },
        },
        "active_ge1_subsample_n": int(ge1_mask.sum()),
    }

    # --- Write outputs -----------------------------------------------------
    write_csv(features)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary


def write_csv(features: list[dict[str, Any]], path: Path = OUT_CSV) -> None:
    """Write per-window features to CSV."""
    fieldnames = [
        "window_id",
        "num_speakers",
        "active_speakers",
        "empty_speakers",
        "silence_fraction",
        "length_ratio",
        "runtime_ratio",
        "always_separated_cpwer",
        "always_mixed_cpwer",
        "hallucinated",
        "mode_s",
        "overlap_label",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for f in features:
            row = {k: f[k] for k in fieldnames}
            writer.writerow(row)


def _format_summary(summary: dict[str, Any]) -> str:
    """Human-readable summary printed to stdout."""
    lines: list[str] = []
    lines.append(f"RQ38 — speaker-count effect on hallucination (n={summary['n_windows']}, "
                 f"hallucinated={summary['n_hallucinated']})")
    lines.append("")
    c = summary["correlations"]
    lines.append("Spearman correlations (active speakers):")
    lines.append(f"  active vs cpWER        : rho={c['active_vs_cpwer']['spearman_rho']:+.4f} "
                 f"(perm p={c['active_vs_cpwer']['perm_p']:.4f})")
    lines.append(f"  active vs hallucination: rho={c['active_vs_hallucination']['spearman_rho']:+.4f} "
                 f"(perm p={c['active_vs_hallucination']['perm_p']:.4f})")
    lines.append(f"  num_speakers vs halluc : rho={c['num_speakers_vs_hallucination']['spearman_rho']:+.4f} "
                 f"(perm p={c['num_speakers_vs_hallucination']['perm_p']:.4f})")
    lines.append(f"  silence_frac vs halluc : rho={c['silence_fraction_vs_hallucination']['spearman_rho']:+.4f} "
                 f"(perm p={c['silence_fraction_vs_hallucination']['perm_p']:.4f})")
    lines.append(f"  length_ratio vs halluc : rho={c['length_ratio_vs_hallucination']['spearman_rho']:+.4f} "
                 f"(perm p={c['length_ratio_vs_hallucination']['perm_p']:.4f})")
    lines.append("")
    lines.append("Hallucination rate by active-speaker stratum (bootstrap 95% CI):")
    for row in summary["stratification_active_speakers"]:
        if row["n"] == 0:
            lines.append(f"  active={row['stratum']}: n=0")
            continue
        lines.append(
            f"  active={row['stratum']}: n={row['n']:2d}  rate={row['rate']:.3f}  "
            f"CI=[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
    lines.append("")
    pc = summary["partial_correlations"]
    lines.append("Partial correlations (active vs hallucination):")
    lines.append(f"  controlling silence_fraction      : {pc['active_vs_halluc_controlling_silence']:+.4f}")
    lines.append(f"  controlling length_ratio           : {pc['active_vs_halluc_controlling_lengthratio']:+.4f}")
    lines.append(f"  controlling silence_frac (act>=1)  : {pc['active_vs_halluc_controlling_silence_active_ge1']:+.4f}")
    lines.append(f"  controlling length_ratio (act>=1)  : {pc['active_vs_halluc_controlling_lengthratio_active_ge1']:+.4f}")
    lines.append(f"  simple rho (act>=1 subsample, n={summary['active_ge1_subsample_n']}): "
                 f"{pc['active_vs_halluc_simple_active_ge1']['spearman_rho']:+.4f} "
                 f"(perm p={pc['active_vs_halluc_simple_active_ge1']['perm_p']:.4f})")
    lines.append("")
    h = summary["hypotheses"]
    lines.append("Hypothesis verdicts:")
    lines.append(f"  H38a: {h['H38a']['verdict']}  (rho={h['H38a']['rho_active_vs_hallucination']:+.4f})")
    lines.append(f"  H38b: {h['H38b']['verdict']}  (mode_s active={h['H38b']['mode_s_active_speaker_counts']})")
    lines.append(f"  H38c: {h['H38c']['verdict']}  (partial_silence={h['H38c']['partial_active_vs_halluc_controlling_silence']:+.4f})")
    return "\n".join(lines)


def main() -> None:
    summary = run_analysis()
    print(_format_summary(summary))
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
