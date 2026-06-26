# RQ45: Multi-Meeting Threshold Stability Simulation

> **Label: `experimental/frontier`** — Builds on RQ13 (PR #904), RQ16 (PR #912),
> RQ25 (PR #929), and RQ44 (PR #963).
> Reanalysis only (no Whisper / no ASR run); reuses the lang-id entropy detector
> from RQ13/RQ16/RQ25/RQ44 verbatim and the existing AISHELL-4 external-validation
> windows (PR #890). Does NOT overwrite any verified reference / gold table.

## Executive Summary

RQ44 (PR #963) showed that bootstrap-resampling the 77 AISHELL-4 windows
produces a **6-modal** lang-id entropy threshold distribution spanning
[0.01, 0.95] (H44b KILLED, interval width 0.94). RQ44 attributed the
6-modality to the small sample size (n=77): the 2 Mode S windows and the
high-entropy clean windows with tied cpWER each pull the "max sensitivity at
≥90% specificity" calibration rule to a different operating point. RQ45 asks
the natural follow-up: **does the threshold distribution converge to unimodal
as the calibration sample size increases?** We simulate a larger calibration
corpus by bootstrap-resampling the 77 AISHELL-4 windows at sample sizes
n ∈ {77, 154, 308, 616, 1232} (1×, 2×, 4×, 8×, 16× the original), with
B=2000 resamples per size, seed=42.

**Two of three pre-registered hypotheses are supported; the picture is
nuanced and the convergence is partial:**

| Hypothesis | Verdict | Test statistic (n=616) | Kill threshold |
|---|---|---:|---|
| H45a: ≤ 2 distinct thresholds with ≥5% frequency (unimodal) | **SUPPORTED** | n_modes = 1 (0.38 at 95.2%) | > 2 modes |
| H45b: 2.5/97.5 percentile interval width < 0.20 | **KILLED** | 0.49 (interval [0.38, 0.87]) | ≥ 0.20 |
| H45c: median OOB cpWER < 1.05 | **SUPPORTED\*** | 1.0000 (but only 45/2000 valid OOB resamples) | ≥ 1.05 |

\* H45c is supported by the letter of the pre-registration but the estimate is
**not meaningful** — at n=616 the expected OOB size is ~0.025 windows, so 97.8%
of resamples have an empty OOB set; the median is computed over the 45 lucky
resamples that happened to leave ≥1 window out. See Honest Limitations.

The headline finding is a **split between two notions of convergence**. The
threshold distribution **does** converge in *modality*: the number of modes
with ≥5% frequency falls 5 → 2 → 2 → 1 → 1 across n = 77 → 154 → 308 → 616 →
1232, and the dominant 0.38 mode grows from 60.9% to 98.6% of the mass. By
n=616 the distribution is unimodal (H45a supported). But the distribution
**does not** converge in *percentile-interval width* at n=616: the 2.5/97.5
interval stays pinned at [0.38, 0.87] (width 0.49) from n=308 through n=616,
because the rare "bad" 0.87 mode persists at ~5–11% frequency — enough to
keep the 97.5th percentile at 0.87 but not enough to count as a ≥5% mode.
The width only collapses to 0 at n=1232, when the 0.87 mode finally drops
below the 2.5% percentile-tail threshold.

The mechanism is that **mode-count convergence and tail-percentile convergence
have different thresholds** (5% vs. 2.5%) and the rare bad mode crosses the
former before the latter. This refines RQ44's conclusion: the 6-modality is
indeed a small-sample artefact that dilutes with larger n, but the
*percentile instability* (RQ44's H44b kill) is more stubborn — it requires the
rare mode to drop below 2.5%, which on this meeting's composition happens
between n=616 and n=1232. The deployable takeaway: **0.38 is the stable
operating point** (≥95% of resamples at n≥616), but a new meeting's
re-calibration still has a ~5% chance of landing on the 0.87 bad mode at
n=616, which is exactly the tail RQ44 flagged as the deployability risk.

## Method

### Data (read-only, not overwritten)

`results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
(label `external/sanity-check`, PR #890): 77 windows of 30 s from AISHELL-4
meeting `M_R003S02C01` (6 speakers, 38.5 min). Each window already stores
`always_mixed_cpwer`, `always_separated_cpwer`, `router_v2_cpwer`,
`oracle_best_cpwer`, and the per-speaker separated transcripts. No ASR is run;
the corrected router's per-window cpWER is the chosen route's stored cpWER.

Hallucination label: `always_separated_cpwer > 1.0` (cpWER > 1.0 means
insertions dominate). This gives 37 hallucinated / 40 clean — matching
RQ12/RQ13/RQ16/RQ25/RQ44.

### Detector (RQ13/RQ16/RQ25/RQ44 verbatim)

`lang_id_entropy(text)`: Shannon entropy (bits) over the Unicode
script-category distribution of the text, computed via `unicodedata.name`.
Clean Chinese is near-monoscript Han (entropy ~ 0); diverse multilingual
gibberish mixing Han+Latin+Katakana+Hangul has high entropy. Per-speaker
scores are aggregated by MAX across the separated tracks (worst-case speaker,
the RQ12/RQ13 convention). The primitive is lifted verbatim from RQ44 so
thresholds are directly comparable.

### Routing rule (RQ13/RQ16/RQ25/RQ44 convention)

HIGH lang-id entropy = diverse gibberish = hallucination. The detector flags
the separated track when `lang_id_entropy >= threshold`:

- if `lang_id_entropy >= threshold` -> flag hallucinated -> route MIXED (cpWER = `always_mixed_cpwer`)
- else -> route SEPARATED (cpWER = `always_separated_cpwer`)

### Calibration rule (RQ44 verbatim)

Sweep threshold over the grid {0.00, 0.01, 0.02, ..., 2.00} (201 candidates).
For each candidate `t`, flag = `score >= t`; compute sensitivity = TP/(TP+FN)
and specificity = TN/(TN+FP) on the calibration set. Select the threshold with
specificity >= 0.90 and maximal sensitivity. Tie-breaker: higher specificity,
then lower threshold (more sensitive). This is RQ44's
`calibrate_threshold_at_spec` exactly, re-implemented as the testable helper
`calibrate_threshold_at_spec(scores, labels, grid, target_spec)`.

### Bootstrap procedure (RQ45)

For each sample size n ∈ {77, 154, 308, 616, 1232} (exact multiples of 77:
1×, 2×, 4×, 8×, 16× the original corpus):

1. Draw B=2000 bootstrap resamples of size n **with replacement** from the 77
   windows (duplicates allowed; n may exceed 77, simulating a larger
   calibration corpus drawn from the same underlying meeting population).
2. On each resample, calibrate the lang-id entropy threshold (≥90%
   specificity, max sensitivity) on the resampled windows (with duplicates
   weighted by their multiplicity in the resample, as is standard for
   bootstrap aggregation).
3. Record the calibrated threshold.
4. Identify the **out-of-bag (OOB) windows** — those NOT drawn in the
   resample — and compute the corrected router's cpWER on the OOB windows at
   the resample's calibrated threshold.

**Seed rule.** n=77 uses seed 42 (directly comparable to RQ44's seed-42
resampling, so the n=77 result is a B=2000 subset of RQ44's B=10000 run).
Larger n use seed `42 + n_sample` so the resamples for different n are
independent (otherwise the n=77 resample would be a prefix of the n=154
resample, coupling them artificially).

**Mode definition.** A "mode" is a distinct threshold value whose bootstrap
frequency is ≥ `MODE_MIN_FRACTION` (5%). `count_modes(thresholds, 0.05)`
returns the number of such modes. H45a's unimodality criterion is
`n_modes <= 2`.

**OOB note.** The expected OOB size is `77 · (1 − 1/77)^n`. For n=77 this is
~28.14 (the classic ~36.8% OOB fraction); for n=308 it is ~1.37; for n=616
it is ~0.025; for n=1232 it is ~8×10⁻⁶. The OOB cpWER evaluation therefore
**becomes undefined for most resamples at n ≥ 308** — this is an inherent
structural limitation of simulating a larger corpus by resampling a fixed
77-window pool (the OOB mechanism assumes n_sample ≈ n_population). The OOB
cpWER is reported per-n but should be read as illustrative for n ≥ 308 and
meaningless for n ≥ 1232.

### Statistics

B=2000 bootstrap resamples per n, seed=42 (n=77) / seed=42+n_sample (larger
n). numpy + stdlib only (no scipy / sklearn / Whisper / meeteval). The
threshold and OOB cpWER distributions are reported as percentiles and mode
tables; no parametric CI is assumed.

## Results

### In-sample reproduction (calibrate + evaluate on all 77 windows)

To verify the script reproduces RQ25/RQ44 before bootstrapping:

| quantity | this study | RQ25/RQ44 reported |
|---|---:|---:|
| threshold | 0.380 | 0.38 |
| sensitivity | 0.9459 (35/37) | 0.946 |
| specificity | 0.9250 (37/40) | 0.925 |
| corrected cpWER | 1.043290 | 1.043 |

The in-sample threshold (0.38) and cpWER (1.043) match RQ25/RQ44 exactly. The
n=77 bootstrap (B=2000, seed=42) median is also 0.38 and the 2.5/97.5
percentile interval width is 0.94 — reproducing RQ44's B=10000 result (the
extremes are reached within the first 2000 seed-42 resamples).

### Per-n threshold distribution

| n | median | mean | std | 2.5/97.5 pct | width | n_modes (≥5%) | n_unique | dominant | dom. % |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 77 | 0.380 | 0.470 | 0.266 | [0.010, 0.950] | 0.940 | 5 | 6 | 0.38 | 60.9% |
| 154 | 0.380 | 0.459 | 0.212 | [0.010, 0.870] | 0.860 | 2 | 6 | 0.38 | 76.9% |
| 308 | 0.380 | 0.433 | 0.156 | [0.380, 0.870] | 0.490 | 2 | 4 | 0.38 | 88.5% |
| 616 | 0.380 | 0.404 | 0.105 | [0.380, 0.870] | 0.490 | 1 | 2 | 0.38 | 95.2% |
| 1232 | 0.380 | 0.387 | 0.058 | [0.380, 0.380] | 0.000 | 1 | 2 | 0.38 | 98.6% |

Mode tables (thresholds with ≥5% frequency, sorted by descending count):

**n=77** (5 modes, 6 unique values): 0.38 (60.9%) · 0.87 (13.5%) · 0.95 (9.3%)
· 0.01 (9.2%) · 0.33 (5.5%). The 0.84 mode (1.9% in RQ44) is below the 5% bar.

**n=154** (2 modes): 0.38 (76.9%) · 0.87 (15.8%). The 0.01, 0.95, 0.33 modes
collapse below 5% — the larger resample dilutes the rare Mode S / clean-tie
compositions.

**n=308** (2 modes): 0.38 (88.5%) · 0.87 (11.0%). Only the two dominant modes
remain.

**n=616** (1 mode): 0.38 (95.2%). The 0.87 mode drops to ~4.8% — below the 5%
mode bar but still above the 2.5% percentile-tail bar, which is why H45b is
killed while H45a is supported.

**n=1232** (1 mode): 0.38 (98.6%). The 0.87 mode drops to ~1.4% — below the
2.5% percentile-tail bar, so the 97.5th percentile finally falls to 0.38 and
the width collapses to 0.

The convergence in **modality** is monotonic and fast: 5 → 2 → 2 → 1 → 1.
The convergence in **percentile-interval width** is non-monotonic in rate: it
falls 0.94 → 0.86 → 0.49 from n=77 to n=308, then **stalls at 0.49 for
n=308–616** (the 0.87 mode keeps the 97.5th percentile pinned at 0.87), then
collapses to 0 at n=1232. The stall is the structural reason H45b (anchored at
n=616) is killed.

### Per-n OOB cpWER distribution

| n | n_valid / 2000 | exp. OOB size | mean OOB size | median | mean | 2.5/97.5 pct | frac < 1.10 |
|---:|---:|---:|---:|---:|---:|---|---:|
| 77 | 2000 | 28.14 | 28.25 | 1.0539 | 1.0680 | [1.000, 1.204] | 0.763 |
| 154 | 2000 | 10.29 | 10.28 | 1.0370 | 1.0616 | [1.000, 1.295] | 0.746 |
| 308 | 1561 | 1.37 | 1.40 | 1.0000 | 1.0536 | [1.000, 1.500] | 0.875 |
| 616 | 45 | 0.025 | 0.023 | 1.0000 | 1.0741 | [1.000, 1.950] | 0.889 |
| 1232 | 0 | 8×10⁻⁶ | 0.000 | — | — | — | — |

The OOB cpWER evaluation **breaks down structurally** at n ≥ 308. At n=77 the
OOB set is the classic ~28-window held-out set (RQ44's regime); the median
OOB cpWER (1.054) reproduces RQ44's 1.056 (the small gap is B=2000 vs.
B=10000). At n=154 the OOB set shrinks to ~10 windows and the median improves
to 1.037 (the larger in-bag calibration set gives a more stable threshold,
which helps the smaller OOB set). At n=308 only 1561/2000 resamples have a
non-empty OOB set (expected size ~1.4), and the median is 1.0 — but the
distribution becomes extremely noisy (max 3.0, 97.5th pct 1.5) because each
OOB set has only ~1 window. At n=616 only 45/2000 resamples have a non-empty
OOB set, and at n=1232 none do.

The H45c verdict (median OOB cpWER < 1.05 at n=616) is technically supported
(median = 1.0000 < 1.05) but the estimate is computed over only 45 resamples
each with a tiny (~1-window) OOB set, so it is **not a meaningful stability
signal** — see Honest Limitations.

### Convergence plots (CSV data)

These are the four requested n-vs-metric series, taken from
`multi_meeting_threshold_results.csv`:

**n vs. median threshold** — flat at 0.38 for all n (the median never moves):

```
n,median_threshold
77,0.3800
154,0.3800
308,0.3800
616,0.3800
1232,0.3800
```

**n vs. percentile interval width** — falls then stalls then collapses:

```
n,interval_width
77,0.9400
154,0.8600
308,0.4900
616,0.4900
1232,0.0000
```

**n vs. number of modes (≥5%)** — monotonic convergence to 1:

```
n,n_modes
77,5
154,2
308,2
616,1
1232,1
```

**n vs. median OOB cpWER** — degrades to undefined (NaN) as OOB empties:

```
n,median_oob_cpwer
77,1.0539
154,1.0370
308,1.0000
616,1.0000
1232,nan
```

## Hypothesis Verdicts

- **H45a — at n=616, threshold distribution is unimodal (≤ 2 distinct
  thresholds with ≥5% frequency): SUPPORTED.** At n=616 there is exactly 1
  mode with ≥5% frequency: threshold 0.38 (count 1904/2000 = 95.2%). The
  second-most-frequent threshold (0.87) is at ~4.8%, just below the 5% mode
  bar. The number of modes falls monotonically 5 → 2 → 2 → 1 → 1 across
  n = 77 → 154 → 308 → 616 → 1232, confirming RQ44's hypothesis that the
  6-modality is a small-sample artefact driven by the 2 Mode S windows and
  the high-entropy clean windows with tied cpWER: as n grows, the rare
  resample compositions that produce the 0.01 / 0.33 / 0.95 modes become
  vanishingly unlikely, and only the dominant 0.38 mode (and, until n=1232,
  the residual 0.87 mode) survive. The 0.38 threshold — RQ25/RQ44's
  in-sample and bootstrap-median value — is the stable operating point.

- **H45b — at n=616, 2.5/97.5 percentile interval width < 0.20: KILLED.**
  The interval is [0.380, 0.870], width 0.49 — 2.45× the 0.20 kill
  threshold. The kill is structural, not a fluke of B=2000: the width stalls
  at exactly 0.49 for both n=308 and n=616 because the rare 0.87 mode
  persists at ~5–11% frequency across that range, keeping the 97.5th
  percentile pinned at 0.87. The width only collapses to 0 at n=1232, when
  the 0.87 mode finally drops below the 2.5% percentile-tail bar (it is at
  ~1.4% at n=1232). The mechanism is that **mode-count convergence (5% bar)
  precedes tail-percentile convergence (2.5% bar)**: H45a is supported at
  n=616 because the 0.87 mode drops below 5%, but H45b is killed at n=616
  because that same mode is still above 2.5%. The honest reading is that
  RQ44's H44b percentile instability is more stubborn than its modality:
  a ~5% chance of calibrating the 0.87 bad mode persists at n=616, which is
  exactly the tail RQ44 flagged as the deployability risk. The pre-registered
  n=616 anchor (8× the original) is insufficient for percentile stability on
  this meeting's composition; n=1232 (16×) is required.

- **H45c — at n=616, median OOB cpWER < 1.05: SUPPORTED\* (estimate not
  meaningful).** The median OOB cpWER over the 45 valid (non-empty-OOB)
  resamples is 1.0000 < 1.05. However, this estimate is **not a meaningful
  stability signal**: at n=616 the expected OOB size is ~0.025 windows, so
  1955/2000 resamples (97.8%) have an empty OOB set and return NaN; the 45
  valid resamples each have a tiny OOB set (~1 window), so their cpWER is
  essentially `min(mixed, separated)` on a single window — heavily
  concentrated at 1.0 (the cpWER-tie value) and extremely noisy (max 2.0,
  97.5th pct 1.95). The H45c "support" is therefore an artefact of the OOB
  mechanism breaking down at large n_sample, not evidence that the router's
  held-out cpWER is genuinely below 1.05. The meaningful OOB signal is at
  n=77 (median 1.054, reproducing RQ44's 1.056) and n=154 (median 1.037);
  beyond that the OOB evaluation cannot be trusted. A proper test of H45c
  would require a genuinely larger corpus (multiple meetings), not
  resampling a fixed 77-window pool.

## Honest Limitations

1. **Single meeting, 77 distinct windows.** This is the central limitation
   and it is *not* resolved by RQ45. The "larger calibration corpus" is
   simulated by drawing n_sample > 77 indices with replacement from the same
   77 AISHELL-4 windows — the underlying population is still 1 meeting
   (M_R003S02C01). Increasing n_sample does not add new information; it only
   dilutes the influence of rare resample compositions. A genuine multi-
   meeting corpus (RQ44's recommended next step, requiring the RQ7 external-
   validation slice work) would add new window *types*, not just new
   duplicates. RQ45's convergence is therefore a convergence *under
   resampling of this meeting*, not a convergence *to a population-level
   threshold*. The 0.38 mode is stable because it is this meeting's
   in-sample threshold; a different meeting could have a different dominant
   mode.

2. **OOB cpWER evaluation breaks down at n ≥ 308.** The expected OOB size
   `77·(1−1/77)^n` falls from 28.14 (n=77) to 1.37 (n=308) to 0.025 (n=616)
   to 8×10⁻⁶ (n=1232). At n=616 only 45/2000 resamples have a non-empty OOB
   set; at n=1232 none do. The OOB mechanism is a held-out evaluation that
   *assumes* n_sample ≈ n_population; when n_sample ≫ n_population, every
   window is drawn at least once with overwhelming probability and there is
   nothing to hold out. H45c's "support" at n=616 is therefore an artefact
   of the OOB breakdown, not a genuine stability signal. The OOB cpWER
   numbers for n ≥ 308 in the results table are reported for completeness
   but should be read as illustrative, not as estimates. A proper held-out
   evaluation at large n requires a train/test split across *distinct*
   meetings.

3. **Duplicates in the resample change the calibration geometry.** When
   n_sample > 77, the resample contains duplicate windows, and the
   calibration rule treats each duplicate as a separate observation (so a
   window drawn 8 times contributes 8× to the sensitivity/specificity
   counts). This is standard bootstrap aggregation, but it means the
   "larger calibration corpus" is not equivalent to a genuinely larger
   independent corpus — the effective number of distinct windows is capped
   at 77, and the calibration rule's discontinuities (at the specificity
   boundary) are smoothed by duplication rather than by new data. The
   convergence in modality is therefore partly a smoothing artefact of
   duplication, not purely evidence that the calibration rule has "more
   data" to work with.

4. **Mode-count convergence vs. percentile-tail convergence are different
   criteria.** H45a (≤2 modes with ≥5%) is supported at n=616 while H45b
   (width <0.20) is killed, because the 5% mode-frequency bar is higher
   than the 2.5% percentile-tail bar. The rare 0.87 mode crosses the former
   (drops below 5%) before the latter (drops below 2.5%). This is not a
   contradiction — it is the correct diagnosis that "unimodal" (in the ≥5%
   sense) does not imply "percentile-stable." The pre-registration treated
   these as related but they are governed by different tail thresholds.
   Future threshold-stability reports should specify which convergence
   criterion is meant; the mode table and the percentile interval should
   both be reported, as RQ44/RQ45 do.

5. **B=2000 (not B=10000).** RQ44 used B=10000; RQ45 uses B=2000 per n to
   keep the 5-n sweep tractable (~10 s total). The n=77 result reproduces
   RQ44's B=10000 median (0.38) and interval width (0.94) exactly because
   the extremes are reached within the first 2000 seed-42 resamples, but
   the mode frequencies for rare modes (e.g. the 0.84 mode at ~1.9%) are
   noisier at B=2000. The H45a verdict (n_modes at n=616) is robust to B
   because the 0.87 mode is at ~4.8% — well away from the 5% boundary in
   expectation but close enough that a different B could flip it. The
   verdict should be read as "n_modes ∈ {1, 2} at n=616 depending on B,"
   with the pre-registered ≤2 criterion met either way.

6. **Calibration rule fixed at "max sensitivity at ≥90% specificity".** A
   different rule (e.g. maximise Youden's J, fix specificity at 95%, or use
   a parametric ROC fit) would produce a different threshold distribution
   and a different convergence rate. The 0.87 mode exists because this
   rule's discontinuous behaviour at the specificity boundary lets some
   resamples "miss" the high-entropy hallucinated windows that 0.38 catches.
   A smoother rule might remove the 0.87 mode entirely, collapsing the
   distribution to the 0.38 mode at smaller n. This is the same caveat as
   RQ44's limitation 5.

7. **cpWER is utterance-level (whole Chinese string = 1 token).** RQ30
   (`results/frontier/meeteval_cpwer_validation/`, PR #935) showed the
   project's cpWER pipeline passes each speaker's full Chinese utterance as
   a single token, so cpWER > 1.0 measures *extra inserted speaker-streams*
   per window, not character-level transcription accuracy. All thresholds,
   OOB cpWERs, and the 1.05/1.10 kill thresholds here are at the utterance
   level. A char-level re-validation (RQ31/RQ35) is the required follow-up
   before claiming the bagged threshold generalises at character granularity.

8. **No deployable routing input from cpWER.** Per the project's hard safety
   rules, cpWER / references are not used as routing input — the lang-id
   entropy detector is computed only from the hypothesis transcripts (the
   deployable signal surface). The hallucination label
   (`always_separated_cpwer > 1.0`) is used only for calibration and OOB
   evaluation, not for routing.

## Reproducibility

- Script:
  `cd /private/tmp/wt-rq45 && /opt/homebrew/bin/python3 results/frontier/multi_meeting_threshold/multi_meeting_threshold_analysis.py`
  (deterministic; numpy + stdlib only; no scipy / sklearn / Whisper /
  meeteval). Runtime ≈ 10 s on macOS for the full 5-n sweep (B=2000 each).
- Tests: `cd /private/tmp/wt-rq45 && /opt/homebrew/bin/python3 -m unittest tests.test_multi_meeting_threshold -v`
  (56 tests; pins `bootstrap_resample`, `calibrate_threshold_at_spec`,
  `count_modes`, `percentile_interval_width`, `expected_oob_size`,
  `out_of_bag_cpwer`, detector primitives, module constants, an in-sample
  reproduction of RQ25/RQ44's 0.38 threshold, an n=77 bootstrap median
  reproduction, and structural convergence-direction tests).
- Outputs:
  - `multi_meeting_threshold_results.csv` — per-n summary table (n_sample,
    expected/mean OOB size, threshold distribution median/mean/std/min/max/
    percentiles/width/n_modes/n_unique/dominant/dominant_fraction, OOB cpWER
    distribution n_valid/median/mean/percentiles/min/max/frac_below_1.10).
    5 rows.
  - `multi_meeting_threshold_results.json` — full summary (in-sample
    reproduction, per-n threshold + OOB distributions with mode tables,
    hypothesis verdicts) plus `per_bootstrap_arrays` (thresholds, oob_cpwer,
    n_oob per n) for reproducibility.
- Bootstrap: B=2000 per n, seed=42 (n=77) / seed=42+n_sample (larger n).
- Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
  (label `external/sanity-check`, read-only — not modified).

## What this changes for the project

RQ44 concluded that the 6-modal threshold distribution is "genuinely a
distribution, not a number" and that multi-meeting calibration is the
required next step. RQ45 partially tests the first half of that claim by
simulating a larger calibration corpus via resampling. The result refines
RQ44's conclusion in three ways:

1. **The modality IS a small-sample artefact (H45a supported).** The number
   of ≥5% modes falls 5 → 2 → 2 → 1 → 1 across n = 77 → 154 → 308 → 616 →
   1232. By n=616 (8× the original corpus) the distribution is unimodal at
   0.38 (95.2% of resamples). RQ44's 6-modality is therefore confirmed as a
   consequence of the small sample size, not a fundamental property of the
   calibration rule — the rare compositions that produce the 0.01 / 0.33 /
   0.95 modes dilute away with larger n. The deployable threshold 0.38 is
   the stable central operating point.

2. **But the percentile instability is more stubborn than the modality
   (H45b killed).** The 2.5/97.5 percentile interval width stalls at 0.49
   (interval [0.38, 0.87]) for n=308 through n=616, because the rare 0.87
   "bad" mode persists at ~5–11% — above the 2.5% percentile-tail bar but
   (at n=616) below the 5% mode bar. The width only collapses to 0 at
   n=1232 (16×). The pre-registered n=616 anchor is therefore sufficient
   for *mode-count* convergence but not for *percentile-tail* convergence.
   The residual ~5% chance of calibrating the 0.87 bad mode at n=616 is
   exactly the deployability tail RQ44 flagged; RQ45 shows it shrinks with
   n but does not vanish by 8×.

3. **The OOB evaluation cannot test held-out cpWER at large n (H45c
   not meaningful).** Simulating a larger corpus by resampling a fixed
   77-window pool empties the OOB set: at n=616 only 45/2000 resamples have
   a non-empty OOB set. The H45c "support" is an artefact of this breakdown,
   not evidence that the router's held-out cpWER is below 1.05. A genuine
   test of held-out cpWER at large calibration sizes requires a multi-
   meeting corpus (RQ44's recommended next step), not resampling.

The concrete next steps RQ45 points to:

1. **0.38 remains the deployable threshold.** RQ45 strengthens RQ44's
   recommendation: 0.38 is not just the bootstrap median at n=77, it is the
   dominant mode at every n from 77 to 1232 (60.9% → 98.6%). Re-calibration
   on a small split remains riskier than deploying 0.38 directly.

2. **Multi-meeting calibration corpus is still the prerequisite.** RQ45's
   convergence is under resampling of *one* meeting; it does not test
   transfer to a new meeting. The RQ7 external-validation slice work
   (staging multiple AISHELL-4 meetings) is the genuine test of whether the
   threshold generalises. RQ45's contribution is to show that *if* the
   threshold distribution is multi-meeting-stable, it will also be
   large-n-stable — but the converse is not established.

3. **Report both mode count and percentile interval.** RQ45 shows these are
   different convergence criteria: a distribution can be unimodal (≥5% bar)
   while still having a wide percentile interval (2.5% bar). Future
   threshold-stability reports should present both, as RQ44/RQ45 do, and
   specify which criterion a hypothesis targets.

4. **Mode S detector (RQ19) remains the complementary fix.** The 0.87 bad
   mode (and the 0.01 mode at small n) exist because the lang-id detector
   cannot distinguish certain hallucination types from clean Chinese on the
   entropy axis. A Mode S detector that catches those windows separately
   would let the lang-id threshold stay at 0.38 and remove the residual
   bad mode, collapsing the percentile interval without needing n=1232.
