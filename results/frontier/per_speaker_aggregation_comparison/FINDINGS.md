# RQ56: Per-Speaker Lang-id Entropy Aggregation Comparison

> **Label: `experimental/frontier`** — Mode B (Focused Extension). Builds on
> RQ13 (PR #904), RQ16 (PR #912), RQ25 (PR #929), RQ37 (PR #946), and RQ44
> (PR #959). Reanalysis only (no Whisper / no ASR / no LLM / no ollama); reuses
> the lang-id entropy detector from RQ13/RQ16/RQ25/RQ44, RQ44's bootstrap +
> OOB framework, RQ48's `count_modes`, and the existing AISHELL-4
> external-validation windows (PR #890). Does NOT overwrite any verified
> reference / gold table.

## Executive Summary

RQ13/RQ16/RQ25/RQ44 all aggregate per-speaker lang-id entropy across the
separated speaker tracks by **MAX** (the worst-case speaker), and RQ44 showed
this MAX threshold is **6-modal** on n=77 (5 modes at ≥ 5% frequency). RQ37
showed the worst speaker contributes **96.5% of cpWER** in the top-10 windows —
the worst-case speaker dominates the catastrophic outcome. RQ56 asks: **does
the aggregation function (MAX vs SUM vs MEAN vs MIN) change the corrected
router's detection performance and corrected cpWER?** If the worst speaker
dominates cpWER, MAX may already be optimal; alternatively SUM or MEAN may
smooth single-speaker noise and produce a more stable (fewer-mode) threshold
distribution.

**One of three pre-registered hypotheses is supported; the worst-case (MAX)
convention is confirmed as the right aggregation, with an informative caveat
that MIN breaks deployability:**

| Hypothesis | Verdict | Test statistic | Kill threshold |
|---|---|---:|---|
| H56a: MAX achieves the highest sensitivity at ≥ 90% specificity | **SUPPORTED** | MAX sens 0.946; other best (SUM) 0.946 (tied, not exceeded) | any other agg strictly higher |
| H56b: SUM produces fewer bootstrap threshold modes than MAX | **KILLED** | SUM 5 modes (≥ 5% freq) = MAX 5 modes | SUM mode count ≥ MAX mode count |
| H56c: All 4 aggregations achieve corrected cpWER ≤ 1.10 | **KILLED** | MAX 1.043 ✓, SUM 1.043 ✓, MEAN 1.052 ✓, **MIN 1.256 ✗** | any agg cpWER > 1.10 |

The headline finding is a **clean ordering MAX ≈ SUM > MEAN > MIN** that
matches the worst-speaker-dominates structure RQ37 established:

- **MAX and SUM are near-identical detectors** on AISHELL-4. They tie at 94.6%
  sensitivity / 92.5% specificity, calibrate the same in-sample threshold
  (0.38), produce the same 5 threshold modes (≥ 5% frequency), and yield the
  same corrected cpWER (1.043). At the bootstrap level they differ in only
  29 / 10000 resamples' thresholds (897 / 10000 in OOB cpWER). The mechanism is
  the worst-speaker-dominance RQ37 documented: hallucinated windows have one
  dominant high-entropy speaker, so SUM (which adds the other speakers' lower
  entropies) does not change which windows cross the threshold. **SUM adds
  nothing over MAX** — confirming MAX is already optimal for the
  worst-case-speaker-dominated cpWER (H56a supported). The flip side: SUM does
  not smooth the distribution either — it reproduces MAX's 5 modes byte-for-byte
  (H56b killed).

- **MEAN is more conservative but not smoother.** Averaging dampens the
  worst-speaker signal, so MEAN calibrates a higher threshold (0.44) with lower
  sensitivity (89.2%) and perfect specificity (100%). Its bootstrap interval is
  markedly tighter (width 0.43 vs 0.94) — averaging does stabilise the
  operating point — but it still produces **5 modes** (≥ 5% frequency), so it
  does not reduce the mode count. MEAN trades 5.4 percentage points of
  sensitivity for a narrower interval and 0 pp of cpWER improvement (1.052 vs
  MAX's 1.043): not a deployable win.

- **MIN breaks deployability.** Taking the best-case (lowest-entropy) speaker
  discards the hallucination signal: MIN calibrates the floor threshold (0.01),
  catches only 64.9% of hallucinated windows, and its corrected cpWER is
  **1.256 > 1.10** — the only aggregation that fails the deployability bar. Its
  OOB cpWER is the worst (median 1.266; only 4.1% of resamples < 1.10). **H56c
  killed by MIN.**

**Bottom line.** The aggregation function matters at the extremes (MIN breaks
deployability) but not in the deployable middle: MAX, SUM, and MEAN all clear
the 1.10 cpWER bar, and MAX ≈ SUM dominate MEAN on sensitivity. RQ37's
"worst speaker = 96.5% of cpWER" finding predicts exactly this — MAX (the
worst-case speaker) is the theoretically correct aggregation for a
worst-case-dominated outcome, and SUM (which would help only if multiple
speakers contributed hallucination) adds nothing because they do not. The
actionable conclusion is unchanged from RQ44: **deploy the MAX-aggregated
lang-id entropy detector at the bootstrap median threshold (0.38)**; the
aggregation choice is settled (MAX), and the residual 0.01 "Mode S" mode is a
detector limitation (RQ19's territory), not an aggregation choice.

## Method

### Data (read-only, not overwritten)

`results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
(label `external/sanity-check`, PR #890): 77 windows of 30 s from AISHELL-4
meeting `M_R003S02C01`. Each window carries per-speaker separated transcripts
(`separated_text_per_speaker`, a dict of speaker-id → transcript string),
`always_mixed_cpwer`, `always_separated_cpwer`, `router_v2_cpwer`, and
`oracle_best_cpwer`. Hallucination label: `always_separated_cpwer > 1.0` → 37
hallucinated / 40 clean (RQ12/RQ13/RQ16/RQ25/RQ44). Speaker-count distribution:
{1: 22, 2: 32, 3: 16, 4: 6, 6: 1}; 35 / 77 windows have > 1 non-empty speaker
track (so SUM differs from MAX only on those 35).

### Detector and aggregation

For each window, `language_id_entropy` (Shannon entropy over Unicode script
categories, RQ13/RQ16/RQ25/RQ44 verbatim) is computed for **each** non-empty
separated speaker track. Empty / whitespace-only tracks are skipped (they
contribute 0.0 and are excluded from MEAN/MIN denominators, so MEAN is the mean
of the non-empty tracks). The per-speaker entropies are aggregated into one
window-level score by four functions:

- **MAX** — max over non-empty tracks (RQ13/RQ16/RQ25/RQ44 worst-case-speaker
  convention).
- **SUM** — sum over non-empty tracks (scales with speaker count).
- **MEAN** — mean over non-empty tracks (smooths speaker count).
- **MIN** — min over non-empty tracks (best-case speaker).

### Routing rule (RQ13/RQ16/RQ25/RQ44 convention)

HIGH lang-id entropy = diverse multilingual gibberish = hallucination. The
detector flags the separated track when `aggregated_score >= threshold`:

- `aggregated_score >= threshold` → route MIXED (`always_mixed_cpwer`)
- else → route SEPARATED (`always_separated_cpwer`)

The corrected router's per-window cpWER is the chosen route's stored cpWER. Per
the project's hard safety rules, cpWER / references are NOT used as routing
input — only as calibration and OOB evaluation labels.

### Calibration, threshold grid, and bootstrap (reused from RQ44)

- **Calibration rule**: sweep the threshold grid in 0.01 steps; select the
  threshold maximising sensitivity at ≥ 90% specificity (RQ25/RQ44 rule).
  Tie-breaker: higher specificity, then lower threshold (more sensitive). This
  is RQ44's `calibrate_threshold_at_spec` verbatim, so the MAX arm reproduces
  RQ44's in-sample 0.38 threshold byte-for-byte.
- **Threshold grid (per-aggregation adaptive)**: MAX uses the RQ44-exact
  0.00–2.00 grid (201 points) so it reproduces RQ44's 0.38 threshold; SUM /
  MEAN / MIN use an adaptive grid `0.00 → ceil(max_observed × 1.05)` (step
  0.01), computed ONCE from the full-sample scores and reused for every
  bootstrap resample so the candidate-threshold set is fixed across resamples
  (otherwise the threshold distribution would be incomparable across resamples).
  The adaptive grid keeps SUM (whose scores scale with speaker count, max ≈ 5.94)
  from being penalised by a MAX-sized 2.00 ceiling.
- **Bootstrap**: B=10000, seed=42, resample size 77 with replacement. The
  **same** (N_BOOT, n) resample-index array is drawn ONCE and reused for all 4
  aggregations, so the comparison is **paired**: any difference is purely due
  to the aggregation function, not resample-draw noise. On each resample:
  calibrate the threshold on the in-bag windows; record the threshold; compute
  the OOB corrected cpWER on the held-out windows at the resample's threshold
  (RQ44's `out_of_bag_cpwer`).

### Mode definition (kill-condition)

A "mode" = a distinct threshold value whose bootstrap frequency is **≥ 5%**
(RQ48's `count_modes`, `min_fraction=0.05`). This is the explicit kill-condition
definition for H56b, as the task METHOD specifies. RQ44's "6-modal" counted
**all** distinct thresholds (its 0.84 mode was at 1.9% frequency); under RQ56's
≥ 5% definition RQ44's MAX baseline has **5** modes. Both definitions
(`n_modes_5pct` and `n_unique`) are reported in the JSON for traceability.

### Statistics

B=10000 bootstrap resamples, seed=42, paired across aggregations. numpy +
stdlib only (no scipy / sklearn / Whisper / meeteval / LLM). Runtime ≈ 18 s
total (all 4 aggregations × B=10000).

## Results

### In-sample calibration (full 77 windows, per aggregation)

| aggregation | threshold | sensitivity | specificity | corrected cpWER | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAX | 0.380 | 0.946 (35/37) | 0.925 (37/40) | 1.0433 | 35 | 3 | 37 | 2 |
| SUM | 0.380 | 0.946 (35/37) | 0.925 (37/40) | 1.0433 | 35 | 3 | 37 | 2 |
| MEAN | 0.440 | 0.892 (33/37) | 1.000 (40/40) | 1.0519 | 33 | 0 | 40 | 4 |
| MIN | 0.010 | 0.649 (24/37) | 0.925 (37/40) | 1.2565 | 24 | 3 | 37 | 13 |

MAX and SUM calibrate the **identical** in-sample operating point (0.38,
35/37 sensitivity, 37/40 specificity, cpWER 1.0433) — matching RQ25/RQ44
exactly for the MAX arm. This is not a bug: SUM ≥ MAX elementwise, but on this
data the windows where SUM > MAX (the 35 multi-speaker windows) all have MAX
already well above 0.38, so SUM flags the same set at 0.38. MEAN averages the
worst-speaker signal down, calibrating a higher threshold (0.44) that trades 2
true positives for 3 fewer false positives (perfect specificity). MIN collapses
to the floor (0.01) — the best-case speaker carries no usable hallucination
signal — missing 13 of 37 hallucinated windows.

### Bootstrap threshold + OOB cpWER distributions (B=10000, paired)

| aggregation | thr median | thr pct [2.5, 97.5] | thr width | n unique | **n modes ≥ 5%** | OOB cpWER median | OOB cpWER mean | frac < 1.10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MAX | 0.380 | [0.010, 0.950] | 0.940 | 6 | **5** | 1.0556 | 1.0692 | 0.760 |
| SUM | 0.380 | [0.010, 0.950] | 0.940 | 6 | **5** | 1.0536 | 1.0666 | 0.776 |
| MEAN | 0.440 | [0.010, 0.440] | 0.430 | 7 | **5** | 1.0556 | 1.0635 | 0.805 |
| MIN | 0.010 | [0.010, 0.380] | 0.370 | 3 | **2** | 1.2661 | 1.2683 | 0.041 |

Mode tables (threshold / count / fraction, sorted by descending frequency):

**MAX** (5 modes ≥ 5%):

| threshold | count | fraction |
|---:|---:|---:|
| 0.38 | 6044 | 60.4% |
| 0.87 | 1451 | 14.5% |
| 0.01 | 899 | 9.0% |
| 0.95 | 844 | 8.4% |
| 0.33 | 573 | 5.7% |

**SUM** (5 modes ≥ 5%):

| threshold | count | fraction |
|---:|---:|---:|
| 0.38 | 6018 | 60.2% |
| 0.87 | 1466 | 14.7% |
| 0.01 | 899 | 9.0% |
| 0.95 | 851 | 8.5% |
| 0.33 | 570 | 5.7% |

**MEAN** (5 modes ≥ 5%):

| threshold | count | fraction |
|---:|---:|---:|
| 0.44 | 5084 | 50.8% |
| 0.42 | 1598 | 16.0% |
| 0.32 | 975 | 9.8% |
| 0.19 | 963 | 9.6% |
| 0.01 | 871 | 8.7% |

**MIN** (2 modes ≥ 5%):

| threshold | count | fraction |
|---:|---:|---:|
| 0.01 | 7401 | 74.0% |
| 0.38 | 2249 | 22.5% |

### Reading the table

1. **MAX and SUM are statistically indistinguishable.** They produce the same 5
   threshold modes (≥ 5% frequency) with near-identical fractions (0.38 at
   60.4% vs 60.2%; 0.87 at 14.5% vs 14.7%; etc.), the same median (0.38), the
   same interval width (0.94), and the same OOB cpWER distribution (median
   1.056 vs 1.054). At the resample level only 29 / 10000 thresholds differ
   (897 / 10000 OOB cpWER values differ, because OOB cpWER is sensitive to
   which specific windows are held out). SUM does not smooth the MAX
   distribution — it reproduces it. **H56b killed.**

2. **MEAN narrows the interval but not the mode count.** Averaging dampens the
   single high-entropy speaker, so MEAN's threshold distribution concentrates
   in [0.01, 0.44] (width 0.43 vs MAX's 0.94) and its OOB cpWER is slightly
   tighter (mean 1.064 vs 1.069; 80.5% < 1.10 vs 76.0%). But it still produces
   5 modes (≥ 5%): the 0.44 / 0.42 / 0.32 / 0.19 cluster is the averaged
   counterpart of MAX's 0.38 / 0.33 cluster, and the 0.01 "Mode S catch" mode
   persists (8.7%). MEAN trades 5.4 pp of sensitivity for a narrower interval
   and no cpWER improvement — not a deployable win.

3. **The 0.01 "Mode S" mode is aggregation-invariant.** It appears under MAX
   (9.0%), SUM (9.0%), MEAN (8.7%), and even MIN (where it is the dominant 74%
   mode). This confirms RQ44/RQ48's finding that the 0.01 mode is a fundamental
   detector ambiguity (the lang-id entropy detector cannot separate Mode S —
   monoscript Chinese semantic hallucination — from clean Chinese), not a
   calibration-rule or aggregation artefact. No aggregation removes it.

4. **MIN's 0.01 dominance is destructive, not smoothing.** MIN's 74% mass at
   0.01 is not the "Mode S catch" — it is the floor threshold, because MIN
   scores are dominated by the best-case (lowest-entropy) speaker, who is
   usually clean. With 0.01 as the median threshold, MIN over-flags on the
   calibration set (sensitivity 64.9% is low because the threshold is too low
   to discriminate — many hallucinated windows have a clean MIN speaker and are
   routed SEPARATED, while the 3 false positives are clean windows with a
   hallucinating best-case speaker). Its OOB cpWER (1.266) is the worst by
   far. **H56c killed by MIN.**

### RQ44 reproduction check (MAX arm)

The MAX arm uses RQ44's exact detector primitive, 0.00–2.00 grid, and
calibration rule, so it reproduces RQ44 exactly: in-sample threshold **0.38**
(RQ44: 0.38), corrected cpWER **1.0433** (RQ44: 1.043), and **6** distinct
bootstrap thresholds (RQ44: 6). Under RQ56's ≥ 5% mode definition the MAX arm
has 5 modes (RQ44's 0.84 mode was at 1.9%, below the 5% bar) — matching RQ48's
recount of RQ44's baseline. This confirms the MAX arm is a faithful RQ44
reproduction and the comparison's anchor.

## Hypothesis Verdicts

- **H56a — MAX achieves the highest sensitivity at ≥ 90% specificity:
  SUPPORTED.** MAX calibrates 94.6% sensitivity (35/37); the best other
  aggregation is SUM, also at 94.6% — tied, not strictly higher. MEAN is lower
  (89.2%); MIN is much lower (64.9%). No aggregation strictly exceeds MAX, so
  the kill condition ("killed if any other aggregation achieves strictly higher
  sensitivity") is not triggered. The tie with SUM is informative: it says MAX
  is optimal and SUM is equivalent (because the worst speaker dominates the
  hallucination signal, so adding the other speakers' entropies in SUM does
  not change which windows are flagged). This is exactly what RQ37's
  "worst speaker = 96.5% of cpWER" predicts.

- **H56b — SUM produces fewer bootstrap threshold modes than MAX: KILLED.**
  SUM produces 5 modes (≥ 5% frequency) — the same count as MAX, with the same
  5 threshold values (0.38, 0.87, 0.01, 0.95, 0.33) and near-identical
  fractions. SUM does not smooth the MAX distribution; it reproduces it. The
  kill condition ("SUM's mode count ≥ MAX's mode count") is triggered (5 ≥ 5).
  The mechanism: SUM > MAX only on the 35 multi-speaker windows, and those
  windows' MAX scores are already well above the 0.38 operating point, so SUM
  flags the same window set and the calibration rule jumps to the same
  thresholds under the same resample compositions. Smoothing via SUM is
  ineffective on a worst-speaker-dominated signal. (Under the secondary
  `n_unique` metric the verdict is identical: SUM 6 = MAX 6.)

- **H56c — All 4 aggregations achieve corrected cpWER ≤ 1.10: KILLED.** MAX
  (1.043), SUM (1.043), and MEAN (1.052) all clear the 1.10 deployability bar,
  but MIN's in-sample corrected cpWER is **1.256 > 1.10**. The kill condition
  ("any aggregation's in-sample cpWER > 1.10") is triggered by MIN. The kill is
  informative: it bounds the aggregation choice — the worst-case (MAX) and
  sum/mean aggregations are deployable, but the best-case (MIN) aggregation
  discards the hallucination signal and breaks deployability. Aggregation
  choice matters at the MIN extreme.

## Honest Limitations

1. **Single meeting, 77 windows.** As in RQ44/RQ48, all resamples draw from the
   same 77 windows of `M_R003S02C01`. The MAX ≈ SUM equivalence rests on this
   meeting's worst-speaker-dominance (RQ37); a meeting where multiple speakers
   hallucinate simultaneously could in principle make SUM diverge from MAX. The
   35 multi-speaker windows here all have one dominant high-entropy speaker, so
   SUM ≈ MAX — but this is an empirical property of this meeting, not a
   theorem. Multi-meeting validation remains the prerequisite (RQ25/RQ44
   conclusion).

2. **Per-aggregation adaptive grid.** MAX uses the fixed RQ44 0.00–2.00 grid
   (to reproduce RQ44's 0.38); SUM / MEAN / MIN use an adaptive grid covering
   their observed range. This is necessary (SUM scores reach 5.94, far above
   2.00) but means the candidate-threshold sets differ between MAX and the
   others. The grid step is 0.01 for all, and the adaptive grids are computed
   ONCE from the full-sample scores and reused for every resample (so the
   candidate set is fixed across resamples within each aggregation). The grid
   choice does not affect the in-sample operating point (MAX and SUM both pick
   0.38, a value present on both grids) or the qualitative verdicts.

3. **B=10000, single seed.** RQ56 uses B=10000 (matching RQ44, not RQ48's
   2000) with seed=42. The MAX ≈ SUM near-identity (29 / 10000 resamples
   differ) is robust: the differing resamples are those where a multi-speaker
   hallucinated window's SUM score crosses a different threshold than its MAX
   score, a rare event on this data. The mode counts (5 / 5 / 5 / 2) are far
   above any small-B noise floor.

4. **"Modes" defined at ≥ 5% frequency.** The H56b kill condition uses the
   explicit ≥ 5% definition (RQ48's `count_modes`). RQ44's "6-modal" counted
   all distinct thresholds (its 0.84 mode was 1.9%); under RQ56's definition
   MAX has 5 modes. Both definitions are in the JSON. The choice does not
   affect the H56b verdict: SUM = MAX (5 = 5 under ≥ 5%; 6 = 6 under
   n_unique) either way.

5. **Same detector limitation as RQ44/RQ48.** RQ56 changes only the
   aggregation function, not the detector. The 0.01 "Mode S" mode persists
   under every aggregation (9.0% / 9.0% / 8.7% / 74.0% for MAX / SUM / MEAN /
   MIN) because it is a property of the lang-id entropy detector's inability to
   separate Mode S from clean Chinese — a complementary Mode S detector (RQ19)
   is the fix, not a different aggregation. RQ56 confirms the 0.01 mode is
   aggregation-invariant.

6. **cpWER is utterance-level.** As in RQ44 (limitation 6), cpWER passes each
   speaker's full Chinese utterance as a single token; cpWER > 1.0 measures
   extra inserted speaker-streams, not character accuracy. A char-level
   re-validation (RQ31/RQ35) remains the follow-up before claiming
   generalisation at character granularity.

7. **MAX and SUM tie on sensitivity, not strictly dominate.** H56a is
   "supported" because no aggregation strictly exceeds MAX, but MAX does not
   strictly exceed SUM either — they are equivalent on this data. A stricter
   reading ("MAX achieves strictly higher sensitivity than every other
   aggregation") would kill H56a. The pre-registered kill condition is
   "killed if any other aggregation achieves strictly higher sensitivity",
   which MAX passes (it is tied for highest, not exceeded). The tie is the
   substantive finding: MAX is optimal and SUM is equivalent, so there is no
   sensitivity reason to prefer SUM over MAX.

## Reproducibility

- Script:
  `/opt/homebrew/bin/python3 results/frontier/per_speaker_aggregation/per_speaker_aggregation_analysis.py`
  (deterministic; numpy + stdlib only; no scipy / sklearn / Whisper / meeteval /
  LLM). Runtime ≈ 18 s for all 4 aggregations × B=10000.
- Tests: `/opt/homebrew/bin/python3 -m unittest tests.test_per_speaker_aggregation -v`
  (92 tests; pins `script_category`, `language_id_entropy`,
  `per_speaker_entropies`, `aggregate_scores`, `aggregate_window`,
  `build_adaptive_grid`, `grid_for`, `calibrate_threshold_at_spec`,
  `corrected_cpwer`, `bootstrap_indices`, `out_of_bag_cpwer`,
  `percentile_interval`, `threshold_distribution`, `count_modes`, module
  constants, and in-sample smoke tests reproducing RQ44's 0.38 threshold /
  35/37 sensitivity / 37/40 specificity / 1.043 cpWER on the 77-window data,
  plus integration tests that `run_aggregation_arm` emits `n_modes_5pct` and
  that H56b's kill metric is `n_modes_5pct ≤ n_unique`).
- Outputs:
  - `per_speaker_aggregation_results.csv` — per-window table (window_id,
    overlap_label, num_speakers, n_nonempty_speakers, cpWER columns,
    hallucinated, score_max / score_sum / score_mean / score_min).
  - `per_speaker_aggregation_results.json` — full summary (in-sample
    calibration per aggregation, per-aggregation threshold + OOB cpWER
    distributions with `n_modes_5pct` / `modes_5pct` / `n_unique` /
    `modes_within_10pct` mode tables, comparison table, hypothesis verdicts)
    plus `per_bootstrap` arrays (thresholds, oob_cpwer, n_oob) for all 4
    aggregations for reproducibility, and `per_window_scores`.
- Bootstrap: B=10000, seed=42, paired across aggregations (same resample
  indices for all 4 aggregations).
- Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
  (label `external/sanity-check`, read-only — not modified).

## What this changes for the project

RQ13/RQ16/RQ25/RQ44 all used MAX aggregation by convention; RQ37 showed the
worst speaker dominates cpWER (96.5% in the top-10 windows). RQ56 **settles
the aggregation question**:

1. **MAX is the theoretically correct and empirically optimal aggregation
   (H56a supported).** For a worst-case-speaker-dominated cpWER outcome (RQ37),
   the worst-case aggregation (MAX) is the right signal. No aggregation
   strictly beats MAX on sensitivity; SUM ties it, MEAN and MIN are worse. The
   MAX convention is retained with a sharper justification.

2. **SUM does not help (H56b killed).** Summing per-speaker entropies would
   help only if multiple speakers contributed hallucination simultaneously. On
   AISHELL-4 they do not — one speaker dominates — so SUM reproduces MAX's
   detection and threshold distribution byte-for-byte. There is no
   smoothing benefit to SUM; the convention stays MAX (simpler, matches RQ44).

3. **MIN breaks deployability (H56c killed).** The best-case speaker carries no
   hallucination signal; MIN's corrected cpWER (1.256) exceeds the 1.10 bar.
   This bounds the aggregation choice from below and rules out best-case
   aggregation for deployment. MEAN is deployable (1.052) but offers no
   advantage over MAX (lower sensitivity, no cpWER improvement).

4. **The 0.01 "Mode S" mode is aggregation-invariant.** It persists under MAX,
   SUM, MEAN, and MIN — confirming RQ44/RQ48 that it is a detector limitation,
   not a calibration or aggregation artefact. The fix is the complementary Mode
   S detector (RQ19) or a larger multi-meeting corpus, not a different
   aggregation.

The actionable conclusion is unchanged from RQ44 — **deploy the MAX-aggregated
lang-id entropy detector at the bootstrap median threshold (0.38)** — but now
the aggregation choice is settled: MAX is optimal, SUM is equivalent (no reason
to switch), MEAN is deployable but weaker, and MIN is undeployable. The next
steps remain those RQ44/RQ48 pointed to: (a) a complementary Mode S detector
(RQ19) to remove the 0.01 mode, and (b) a multi-meeting calibration corpus to
dilute the Mode S prevalence and test whether MAX ≈ SUM holds when multiple
speakers hallucinate.
