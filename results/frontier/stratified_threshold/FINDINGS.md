# RQ49: Speaker-Count Stratified Threshold for Reduced Modality

> **Label: `experimental/frontier`** — reanalysis-only test of whether stratifying the
> lang-id entropy threshold by active-speaker count reduces the 6-modality that RQ44
> (PR #963) observed in the pooled bootstrap threshold distribution. No Whisper / no
> ASR model is run; this reads the existing AISHELL-4 external-validation results and
> re-runs the RQ44 bootstrap recipe separately on two speaker-count strata. Builds on
> RQ13 (PR #904), RQ16 (PR #912), RQ25 (PR #929), RQ38 (PR #948), and RQ44 (PR #963).
>
> Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
> (label `external/sanity-check`, PR #890).

## Executive Summary

Stratifying the corrected router's lang-id entropy threshold by active-speaker count
(≤2 vs >2) does **not** meaningfully reduce the threshold modality or improve the
out-of-sample cpWER. RQ44's pooled distribution had 6 unique thresholds (5 with ≥5%
frequency) and a median OOB cpWER of 1.056. Stratification removes exactly **one** of
the five modes from the ≤2-speaker stratum (the `0.95` mode, which was driven by the
>2-speaker windows) but leaves **4 modes** in that stratum — failing the H49a bar of
≤3. The >2-speaker stratum is bimodal (H49b supported), but the bimodality is a
degenerate small-sample artifact (15 windows, 1 clean), not a meaningful operating-
point reduction. The combined stratified router's median OOB cpWER (1.055556) is
**numerically identical** to RQ44's exact pooled median — H49c is "supported" only by a
sub-0.001 rounding margin (1.055556 < 1.056) with **no substantive improvement**.

**H49a (≤2-speaker modes ≤ 3): KILLED (4 modes). H49b (>2-speaker modes ≤ 2): SUPPORTED
(2 modes, degenerate). H49c (combined OOB cpWER < 1.056): technically SUPPORTED but
substantively a tie (no improvement).**

The scientifically useful result is the *negative* one: the bulk of RQ44's modality is
**intrinsic to the ≤2-speaker stratum** (where Mode S lives), not separable by speaker-
count stratification. RQ44's hypothesis that the modes are "partly driven by the 2 Mode
S windows (which occur only with ≤2 speakers)" is refined: the `0.95` mode was driven
by >2-speaker windows (it disappears when they are removed), but the Mode-S-driven
`0.01` mode (15.25% in stratum 1) and the `0.38`/`0.87`/`0.33` modes all persist in the
≤2-speaker stratum. Speaker count is therefore **not** an effective stratification lever
for stabilising the threshold; a future RQ should stratify by a feature that actually
separates Mode S from the other operating points (e.g. the per-window silence-gap
fraction or the separated-track length ratio, both available at the transcript level).

## Method

1. **Active speakers** = count of non-empty `separated_text_per_speaker` values (RQ38's
   definition, verbatim). This is the speaker-count proxy RQ38 showed carries the
   hallucination effect (Spearman ρ = +0.611; rate 0% → 34.5% → 65% → 93.3% across
   0 → 1 → 2 → 3+ active speakers).
2. **Stratify** the 77 windows at ≤2 (stratum 1, where Mode S occurs) vs >2 (stratum 2).
3. **Per stratum**: B = 2000 bootstrap resamples (seed = 42, n = stratum_size with
   replacement). On each resample: calibrate the lang-id entropy threshold maximising
   sensitivity at ≥90% specificity (RQ13/RQ16/RQ25/RQ44 rule, verbatim) on the in-bag
   windows; evaluate the corrected-router cpWER on the out-of-bag windows.
4. **Stratified router**: on each aligned bootstrap resample, calibrate a threshold for
   BOTH strata, then evaluate the COMBINED OOB cpWER — each OOB window is routed by its
   OWN stratum's threshold. The combined OOB pools the OOB windows from both strata.

Detector primitives (`script_category`, `language_id_entropy`, `max_across_speakers`)
and the calibration rule (`calibrate_threshold_at_spec`) are lifted **verbatim** from
RQ44 so thresholds are directly comparable. numpy + stdlib only (no scipy / sklearn /
Whisper). A within-script pooled bootstrap at matched B=2000, seed=42 reproduces RQ44's
6 unique / 5 modes (≥5%) / width 0.94, confirming the comparison is apples-to-apples.

## Stratification

| Stratum | Rule | n | Hallucinated | Hallucination rate |
|---|---|--:|--:|---:|
| 1 | active_speakers ≤ 2 | 62 | 23 | 37.1% |
| 2 | active_speakers > 2 | 15 | 14 | 93.3% |

The two Mode S windows (w22, w30 — the low-entropy hallucinated residuals from RQ19/RQ38)
both fall in stratum 1, as expected (H38b: Mode S occurs only with ≤2 active speakers).
The stratum-2 rate (93.3%) matches RQ38's highest active-speaker bin exactly, confirming
the active-speaker definition is the faithful one.

## In-sample reproduction (RQ44 reference)

| Calibration | Threshold | Sensitivity | Specificity | cpWER |
|---|---:|---:|---:|---:|
| Pooled (all 77) | 0.3800 | — | — | 1.043290 |
| Stratum 1 (≤2) | 0.3800 | 0.9130 | 0.9487 | — |
| Stratum 2 (>2) | 0.9500 | 0.8571 | 1.0000 | — |

The pooled in-sample threshold (0.38) and cpWER (1.043290) reproduce RQ44/RQ25 exactly.
Stratum 1's in-sample threshold is also 0.38 (Mode S lives here); stratum 2's is 0.95
(the single clean window's entropy sits just below 0.95, so the ≥90%-specificity rule
pins the threshold there).

## Threshold distributions (B=2000, seed=42)

### Pooled (matched B=2000; RQ44 published B=10000 reference in parens)

| Statistic | Pooled (B=2000) | RQ44 (B=10000) |
|---|---:|---:|
| median | 0.3800 | 0.38 |
| 2.5 / 97.5 pct | [0.0100, 0.9500] | [0.01, 0.95] |
| width | 0.9400 | 0.94 |
| n_unique | 6 | 6 |
| modes (≥5%) | 5 | 5 |
| OOB cpWER median | 1.053922 | 1.055556 |

Pooled modes (≥5%): `0.38` (60.85%), `0.87` (13.45%), `0.95` (9.30%), `0.01` (9.20%),
`0.33` (5.50%).

### Stratum 1 (≤2 active speakers)

| Statistic | Value |
|---|---:|
| median | 0.3800 |
| 2.5 / 97.5 pct | [0.0100, 0.8700] |
| width | 0.8600 (vs pooled 0.94) |
| n_unique | 5 |
| **modes (≥5%)** | **4** |
| OOB cpWER median | 1.062500 |
| OOB cpWER mean | 1.071017 |
| OOB frac < 1.10 | 0.7450 |

Stratum-1 modes (≥5%): `0.38` (66.35%), `0.01` (15.25%), `0.87` (10.55%), `0.33`
(7.10%). **The pooled `0.95` mode (9.30%) disappears** — it was driven by the
>2-speaker windows. The width narrows slightly (0.86 vs 0.94). But 4 modes remain,
exceeding the H49a bar of ≤3.

### Stratum 2 (>2 active speakers)

| Statistic | Value |
|---|---:|
| median | 0.9500 |
| 2.5 / 97.5 pct | [0.0000, 0.9500] |
| width | 0.9500 |
| n_unique | 2 |
| **modes (≥5%)** | **2** |
| OOB cpWER median | 1.000000 |
| OOB cpWER mean | 1.030040 |
| OOB frac < 1.10 | 0.9245 |

Stratum-2 modes (≥5%): `0.95` (64.70%), `0.00` (35.30%). The bimodality is a
**degenerate small-sample artifact**: with 15 windows and only 1 clean window, ~35.3%
of resamples draw 0 clean windows in-bag (≈ the theoretical 1/e ≈ 36.8%), at which
point n_neg = 0, specificity is trivially 1.0 for every threshold, and the tie-breaker
collapses to the lowest threshold (`0.00` = flag everything as MIXED). When the clean
window IS in-bag (~64.7%), the threshold pins to `0.95` (just above the clean window's
entropy). So stratum 2's "2 modes" reflect "clean window in-bag or not", not a genuine
reduction in operating-point ambiguity.

## Combined stratified-router OOB cpWER

| Statistic | Combined (stratified) | RQ44 pooled (B=10000) |
|---|---:|---:|
| n valid resamples | 2000 / 2000 | 10000 |
| **median** | **1.055556** | **1.055556** |
| mean | 1.063161 | — |
| 2.5 / 97.5 pct | [1.0000, 1.1667] | — |
| frac < 1.10 | 0.8070 | 0.7604 |
| frac < RQ44 (1.056) | 0.5115 | — |

The combined stratified router's median OOB cpWER (**1.055556**) is **numerically
identical** to RQ44's exact pooled median (1.055556) and slightly *worse* than the
matched-B pooled median (1.053922). Stratification provides **no substantive OOB cpWER
improvement**. The combined distribution does shift a little more mass below 1.10
(80.70% vs RQ44's 76.04%), but the central tendency is unchanged.

## Hypothesis Verdicts

| Hypothesis | Statement | Result | Kill | Supported? |
|---|---|---:|---|---|
| H49a | ≤2-speaker threshold modes (≥5%) ≤ 3 | 4 | > 3 | **KILLED** |
| H49b | >2-speaker threshold modes (≥5%) ≤ 2 | 2 | > 2 | **SUPPORTED** (degenerate) |
| H49c | combined OOB cpWER < 1.056 | 1.055556 | ≥ 1.056 | **SUPPORTED** (rounding tie; no substantive gain) |

## Interpretation

1. **Speaker-count stratification is not an effective lever for threshold stability.**
   Removing the >2-speaker windows eliminates exactly one of RQ44's five modes (the
   `0.95` mode) from the ≤2-speaker stratum, but four modes remain — the H49a bar of
   ≤3 is not met. The modality is intrinsic to the ≤2-speaker stratum, where Mode S
   lives.

2. **RQ44's "Mode S drives the modality" hypothesis is refined.** The `0.95` mode was
   actually driven by the >2-speaker windows (it disappears when they are removed), not
   by Mode S. Conversely, the Mode-S-driven `0.01` mode (15.25% in stratum 1) and the
   `0.38`/`0.87`/`0.33` modes all persist in the ≤2-speaker stratum. So Mode S drives
   ONE mode (`0.01`); the other three are driven by the within-stratum-1 operating-
   point ambiguities that speaker count cannot separate.

3. **Stratum 2's bimodality is a small-sample artifact**, not a meaningful operating-
   point reduction. With 15 windows and 1 clean window, the two modes are "clean window
   in-bag (threshold 0.95)" vs "clean window OOB (threshold 0.00, flag everything)".
   H49b is technically supported but should not be read as evidence that the >2-speaker
   threshold is stable.

4. **No deployment benefit.** The combined stratified router's median OOB cpWER
   (1.055556) is identical to RQ44's pooled median. Stratifying by speaker count does
   not move the central tendency because the costly hallucination windows are
   concentrated in stratum 1 (where Mode S lives), and a stratum-1 threshold that is
   still 4-modal does not route them more reliably than the pooled 5-modal threshold.

## Caveats

- **B=2000 per stratum** (vs RQ44's B=10000 pooled) to keep runtime reasonable. The
  within-script pooled bootstrap at B=2000 reproduces RQ44's 6-unique/5-modes/width-0.94
  exactly, so the matched-B comparison is valid; the cited RQ44 B=10000 numbers are
  reported alongside.
- **Active-speaker definition** (RQ38) is used rather than the configured `num_speakers`
  field. The two give different stratum sizes (62/15 vs 54/23) but the same qualitative
  conclusion (Mode S in stratum 1). The active-speaker definition is the one RQ38 showed
  carries the hallucination effect (ρ = +0.611), and its stratum-2 rate (93.3%) matches
  the RQ49 spec's quoted figure.
- **H49c's "supported" verdict is a rounding-boundary technicality.** The combined
  median (1.055556) is < 1.056 only because 1.056 is the rounded form of RQ44's exact
  1.055556. Against RQ44's exact pooled median the result is a tie, and against the
  matched-B pooled median (1.053922) it is slightly worse. There is no substantive
  improvement.

## Recommendation for future RQs

Speaker count does not separate Mode S from the other ≤2-speaker operating points. A
more promising stratification lever — available at the transcript level — is the
per-window **silence-gap fraction** or the **separated/mixed length ratio**, both of
which RQ12/RQ38 identified as the audio-level stimulus that speaker count proxies for.
A future RQ could stratify the threshold on one of those features directly, which may
isolate Mode S into its own stratum and yield a genuinely unimodal threshold in the
complement stratum.

## Reproducibility

- Script: `results/frontier/stratified_threshold/stratified_threshold_analysis.py`
- Results: `results/frontier/stratified_threshold/stratified_threshold_results.json`
  (full distributions + per-bootstrap arrays), `stratified_threshold_results.csv`
  (per-stratum summary).
- Tests: `tests/test_stratified_threshold.py` (51 tests; pins
  `stratify_by_speaker_count`, `calibrate_threshold_at_spec`, `bootstrap_stratified`,
  `combined_oob_cpwer`; two smoke tests on the real 77-window data).
- Run: `/opt/homebrew/bin/python3 results/frontier/stratified_threshold/stratified_threshold_analysis.py`
- Tests: `/opt/homebrew/bin/python3 -m unittest tests.test_stratified_threshold -v`
- Seed: 42. B=2000 per stratum. numpy + stdlib only.
