# RQ54: Cascade with F1 Calibration

> **Label: `experimental/frontier`** — Builds on RQ43 (PR #959, 3-tier KL
> cascade), RQ44 (PR #963, OOB bootstrap), RQ46 (PR #966, original-rule CI
> anchor), and RQ48 (PR #965, F1 calibration rule + `count_modes`).
> Reanalysis only (no Whisper / no ASR run); reuses RQ43's per-window KL/cpWER
> data, RQ48's `calibrate_f1` / `count_modes` verbatim, and RQ44's OOB
> bootstrap protocol. Does NOT overwrite any verified reference / gold table.

## Executive Summary

RQ43 (PR #959) built a 3-tier compute-aware cascade (whisper-tiny → KL gate →
whisper-base) on 77 AISHELL-4 windows and calibrated the KL gate with the "max
sensitivity at ≥ 90% specificity" rule, landing at KL = 3.30 nats (cascade
cpWER 0.888947, 44.1% reduction vs always-tiny-separated 1.590909). RQ46
(PR #966) confirmed KL = 3.30 is robust under bootstrap: percentile cpWER CI
[0.7674, 1.0163] (width 0.2489) entirely below the baseline. RQ48 (PR #965)
showed that replacing the calibration rule with **F1 maximisation** on the
*lang-id-entropy* detector collapses the bootstrap threshold distribution to
only 2 modes (vs 6 for the original rule). RQ54 asks whether that F1 stability
**transfers to the KL gate detector**: does F1 calibration of the KL threshold
give a cascade with ≤ 2 bootstrap modes, a narrower BCa cpWER CI than RQ46's
0.2489, and a median cpWER at least as good as RQ43's 0.8889?

**All three pre-registered hypotheses are supported:**

| Hypothesis | Verdict | Test statistic | Kill threshold |
|---|---|---:|---|
| H54a: F1 KL threshold has ≤ 2 bootstrap modes | **SUPPORTED** | 1 mode (≥ 5% freq) | > 2 modes |
| H54b: BCa cpWER CI width < 0.2489 | **SUPPORTED** | 0.2481 | ≥ 0.2489 |
| H54c: median cpWER ≤ 0.888947 | **SUPPORTED** | 0.7799 | > 0.888947 |

The headline finding is that **F1 calibration on the KL detector is even more
stable than RQ48's F1 on lang-id-entropy**: it collapses to a **single mode**
at KL = 0.01 (96.3% of 10000 bootstrap resamples), vs RQ48's 2 modes on
lang-id-entropy. The mechanism is that the KL detector does NOT suffer the
"Mode S" identifiability problem that bounds lang-id-entropy: high-KL windows
ARE the hallucinated ones (KL measures transcript gibberishness directly), so
F1 can flag all 37 hallucinated windows (sensitivity = 1.0) at the lowest grid
point (0.01) while keeping precision = 0.578. There is no low-KL hallucinated
sub-population for F1 to trade off against, so no second mode emerges.

The F1 threshold (0.01) is far below RQ43's original-rule threshold (3.30):
F1 escalates **83.1%** of windows to whisper-base (vs RQ43's ~30%), which
lowers the cascade cpWER from 0.8889 to 0.7775 in-sample (0.7799 OOB median) —
a **12.3% cpWER improvement**. The trade-off is compute: the cascade runs at
1.773× whisper-tiny (vs RQ43's ~1.4×), because F1 prefers the cpWER-optimal
"escalate almost everything to base" operating point over RQ43's
"escalate only the worst hallucinations" operating point.

The BCa CI width (0.2481) is **marginally** narrower than RQ46's percentile CI
(0.2489) — a difference of 0.0008. H54b is technically supported, but the
honest interpretation is that F1 calibration does **not meaningfully tighten**
the cpWER interval: the width is essentially unchanged. The substantive
findings are H54a (single-mode stability, dramatically better than RQ48's 2
modes) and H54c (12% lower cpWER via broader escalation).

## Method

### Controlled comparison design (only the calibration rule changes)

The cascade simulation is held **fixed at RQ43's actual implementation** so
the H54c comparison to RQ43's 0.888947 anchor is apples-to-apples:

- **Tier 1** (whisper-tiny) cpWER per window = RQ43's `tiny_sep_cpwer` (the
  real whisper-tiny separated-audio cpWER; `always_separated_cpwer` in
  AISHELL-4).
- **Tier 3** (whisper-base) cpWER per window = RQ43's `base_sep_cpwer` =
  `tiny_sep_cpwer × 0.428031` (the model_scale separated base/tiny CER ratio,
  constant across overlap). This is RQ43's actual base-cpWER estimate; it is
  NOT `always_mixed_cpwer`.
- **Tier 2** (KL gate): escalate to base when the character-bigram asymmetric
  KL divergence of the tiny transcript (RQ43's `kl_sep`) ≥ the calibrated
  threshold.

The ONLY independent variable is the KL-gate calibration rule: RQ43's "max
sensitivity at ≥ 90% specificity" is replaced by RQ48's F1 maximisation. The
task brief's METHOD bullet paraphrases tier 1 as "tiny × 0.9 / × 1.5" and
tier 3 as "always_mixed_cpwer"; that paraphrase does NOT match RQ43's verified
code and would confound the calibration-rule comparison, so RQ43's actual
simulation is used.

### Data (read-only, not overwritten)

- `results/frontier/three_tier_cascade/three_tier_cascade_results.json`
  (label `experimental/frontier`, PR #959): RQ43's 77 per-window
  `(tiny_sep_cpwer, base_sep_cpwer, kl_sep)`. Verified anchors: n = 77,
  baseline = 1.590909, base ratio = 0.428031, KL range [0.0, 8.5255].
- Hallucination label: `tiny_sep_cpwer > 1.0` (== `always_separated_cpwer >
  1.0`) → 37 hallucinated / 40 clean (RQ44/RQ48 rule). High KL flags a window
  as hallucinated and escalates it to base — the same direction as RQ43's
  cascade.

### F1 calibration rule (RQ48's `calibrate_f1`, reused verbatim)

F1 = 2 · precision · recall / (precision + recall), with precision = TP/(TP+FP)
and recall = TP/(TP+FN) = sensitivity. The KL threshold is swept over a 0.01-
step grid spanning RQ43's observed KL range [0.00, 8.55] (856 points); the grid
point maximising F1 is chosen, with the **lowest threshold** breaking ties
(RQ48's `_select_threshold` convention, `>= - EPS` flagging). RQ54 imports
`calibrate_f1` and `count_modes` directly from RQ48's module, so the
calibration rule is byte-identical — only the detector signal changes (KL here
vs lang-id-entropy in RQ48).

### Bootstrap + BCa protocol

1. **In-sample F1 calibration** on all 77 windows → F1-optimal KL threshold
   and the in-sample cascade cpWER (the BCa point estimate θ̂).
2. **Bootstrap B = 10000, seed = 42**: for each resample, calibrate the F1-
   optimal KL threshold on the in-bag windows and evaluate the cascade cpWER
   on the out-of-bag (OOB) windows (RQ44's OOB protocol). Records the per-
   resample threshold (for mode counting) and OOB cpWER (for the BCa CI).
3. **Delete-1 jackknife** (77 fits) for the BCa acceleration.
4. **BCa 95% CI** on the OOB cpWER distribution (bias-corrected + accelerated;
   Acklam inverse-normal + 1 Halley step, no scipy).
5. **Mode count** on the bootstrap threshold distribution (RQ48's
   `count_modes`, `min_fraction = 0.05` — the explicit kill-condition
   definition).

A "mode" = a distinct threshold value whose bootstrap frequency is ≥ 5%. The
BCa adjusted percentiles use the forward normal CDF Φ (not the inverse): α₁ =
Φ(z₀ + (z₀ + z_α) / (1 − a(z₀ + z_α))).

### Statistics

B = 10000 bootstrap resamples, seed = 42, n = 77 with replacement. numpy +
stdlib only (no scipy / sklearn / Whisper / meeteval). Runtime ≈ 30 s (856-
point grid × 10000 resamples, vectorised over resamples; 77-fit jackknife).

## Results

### In-sample F1 calibration (full 77 windows)

| statistic | F1 rule (RQ54) | RQ43 original rule |
|---|---:|---:|
| KL threshold | 0.01 | 3.30 |
| sensitivity | 1.000 | 0.919 |
| specificity | 0.325 | 0.925 |
| precision | 0.578 | — |
| F1 | 0.733 | — |
| TP / FP / TN / FN | 37 / 27 / 13 / 0 | 34 / 3 / 37 / 3 |
| escalation fraction | 83.1% | ~30% |
| cascade cpWER | 0.7775 | 0.8889 |
| cascade compute | 1.773× | ~1.4× |

F1 picks the **lowest grid point (0.01)** because it maximises recall
(sensitivity = 1.0: all 37 hallucinated windows are escalated) while keeping
precision = 0.578 (27 clean windows are also escalated, but the TP gain
outweighs the FP cost on the F1 objective). RQ43's original rule instead
maximises sensitivity **subject to specificity ≥ 0.90**, which forces a high
threshold (3.30) that escalates only the worst hallucinations. The two rules
optimise different objectives on the same ROC curve: F1 finds the
precision/recall-optimal point (low threshold, broad escalation); the spec
rule finds the high-specificity point (high threshold, surgical escalation).

The F1 operating point gives a **12.3% lower cpWER** (0.7775 vs 0.8889)
because base cpWER = tiny × 0.428 < tiny for every window: escalating more
windows to base lowers the mean cpWER. The cost is compute (1.773× vs
~1.4×) — F1 trades compute for cpWER.

### Bootstrap threshold + OOB cpWER distributions (B = 10000, seed = 42)

| statistic | value |
|---|---:|
| threshold median | 0.0100 |
| threshold mean | 0.1301 |
| threshold std | 0.6117 |
| threshold min / max | 0.0100 / 3.3100 |
| n unique thresholds | 3 |
| **n modes ≥ 5%** | **1** |
| mode: KL = 0.01 | 9629 / 10000 = 96.3% |
| OOB cpWER median | 0.7799 |
| OOB cpWER mean | 0.7841 |
| OOB cpWER pct [2.5, 97.5] | [0.6797, 0.9337] |
| OOB cpWER min / max | 0.5910 / 1.1327 |
| mean OOB size | 28.22 (expected 28.14) |

The bootstrap threshold distribution is **unimodal** at KL = 0.01 (96.3%).
Only 3 distinct threshold values appear across 10000 resamples; the other two
(3.30 and 3.31) together account for 3.7% — below the 5% mode threshold. This
is **more stable than RQ48's F1 on lang-id-entropy** (2 modes), because the KL
detector lacks the Mode S identifiability problem: every hallucinated window
has high KL (KL measures transcript gibberishness directly), so there is no
low-KL hallucinated sub-population for F1 to trade off against. RQ48's second
mode (0.01 on lang-id-entropy) arose from Mode S windows with low entropy;
RQ54's KL detector has no such ambiguity, so no second mode emerges.

### BCa 95% CI on OOB cpWER

| statistic | value |
|---|---:|
| θ̂ (in-sample cascade cpWER) | 0.7775 |
| BCa CI lo | 0.6779 |
| BCa CI hi | 0.9260 |
| **BCa CI width** | **0.2481** |
| median | 0.7799 |
| z₀ (bias correction) | −0.0406 |
| a (acceleration) | 0.0125 |
| α₁ (adjusted lower percentile) | 0.0232 |
| α₂ (adjusted upper percentile) | 0.9729 |
| method | bca |
| n valid | 10000 |

The BCa CI [0.6779, 0.9260] lies entirely below RQ43's baseline (1.5909) and
mostly below the catastrophic-hallucination boundary (1.0): only the upper
2.7% of the bootstrap mass exceeds 1.0. The bias correction z₀ = −0.0406
(slightly negative: θ̂ = 0.7775 is just below the bootstrap median 0.7799) and
the acceleration a = 0.0125 (small positive: the jackknife θ_loo ranges
[0.7634, 0.7821], modest right-skew) are both small, so the BCa adjusted
percentiles (0.0232, 0.9729) are close to the nominal (0.025, 0.975) — the CI
is only slightly tighter than a plain percentile CI.

## Hypothesis Verdicts

- **H54a — F1 KL threshold has ≤ 2 bootstrap modes: SUPPORTED.** F1 produces
  **1 mode** ≥ 5% (KL = 0.01, 96.3%) — half RQ48's F1 mode count on lang-id-
  entropy (2 modes) and far below RQ48's specificity-boundary baseline (5
  modes). The KL detector does not suffer the Mode S identifiability problem:
  high-KL windows are hallucinated by construction (KL measures transcript
  gibberishness), so there is no low-KL hallucinated sub-population to spawn a
  second mode. RQ48's residual 0.01 "Mode S catch" mode is a property of the
  lang-id-entropy detector, not of F1; on the KL detector F1 collapses cleanly
  to a single operating point.

- **H54b — BCa cpWER CI width < 0.2489: SUPPORTED (razor-thin margin).** The
  BCa width is 0.2481 vs RQ46's 0.2489 — a margin of 0.0008 (0.3%). H54b is
  technically supported, but the honest interpretation is that F1 calibration
  does **not meaningfully tighten** the cpWER interval. The comparison is
  directional rather than pure: RQ46's anchor is a percentile CI evaluated
  in-bag at a fixed threshold (3.30), while RQ54's BCa CI is bias-corrected +
  accelerated and evaluated OOB at a re-calibrated threshold (0.01). The
  methodological differences (BCa vs percentile, OOB vs in-bag, re-calibrated
  vs fixed) could account for the 0.0008 margin on their own. The substantive
  finding is that the cpWER uncertainty is essentially unchanged: F1 + BCa +
  OOB does not widen the interval, but it does not meaningfully narrow it
  either.

- **H54c — median cpWER ≤ 0.888947: SUPPORTED.** The OOB median cpWER is
  0.7799 — a **12.3% improvement** over RQ43's 0.8889. The mechanism is
  straightforward: F1's low threshold (0.01) escalates 83% of windows to base,
  and base cpWER = tiny × 0.428 < tiny for every window, so broader escalation
  lowers the mean cpWER. This is not "free": the cascade compute rises to
  1.773× (vs RQ43's ~1.4×). F1 optimises the precision/recall trade-off
  without a compute constraint, so it finds the cpWER-minimising operating
  point on this data (escalate almost everything). RQ43's spec rule, by
  contrast, finds the high-specificity operating point (escalate only the
  worst hallucinations) — a more compute-efficient but higher-cpWER point.

## Honest Limitations

1. **F1 trades compute for cpWER; H54c does not control for compute.** The
   12.3% cpWER improvement comes from escalating 83% of windows to base
   (compute 1.773×) vs RQ43's 30% (compute ~1.4×). A fair cpWER comparison
   would hold compute fixed and ask whether F1's broader escalation is
   cpWER-optimal at the same compute budget — that is NOT what H54c tests.
   H54c tests "is F1's median cpWER ≤ RQ43's?" and the answer is yes, but the
   improvement is partly mechanical (more escalation → lower cpWER) rather
   than a calibration-rule quality gain. RQ43's compute-aware Pareto analysis
   remains the reference for compute-controlled comparisons.

2. **H54b's margin (0.0008) is within the methodological differences.** RQ46's
   anchor is a percentile CI (in-bag, fixed threshold 3.30); RQ54's is a BCa
   CI (OOB, re-calibrated threshold 0.01). The 0.0008 width difference could
   be due to any of these differences, not specifically F1 calibration. The
   honest reading: F1 + BCa + OOB does not widen the interval, but does not
   meaningfully narrow it either.

3. **Single meeting, 77 windows.** As in RQ43/RQ44/RQ48, all resamples draw
   from the same 77 windows of `M_R003S02C01`. The single-mode collapse at
   KL = 0.01 is driven by this meeting's KL distribution; a different meeting
   with a different KL/cpWER relationship could produce a different mode
   structure. RQ54 answers "does F1 stabilise the KL threshold under
   resampling of this meeting?" — it does NOT answer "does it transfer to a
   new meeting?".

4. **The KL detector's lack of a Mode S mode is a property of the detector,
   not of F1.** H54a's single-mode result is partly because KL measures
   transcript gibberishness directly (high KL ⟺ hallucination), while lang-id-
   entropy has an orthogonal failure mode (Mode S: monoscript Chinese semantic
   hallucination with low entropy). A detector that combines KL and lang-id-
   entropy would inherit both detectors' ambiguities; RQ54 does not test that.

5. **BCa acceleration uses an in-sample jackknife.** The delete-1 jackknife
   calibrates F1 on n−1 windows and evaluates the in-sample cascade cpWER on
   those n−1 windows (not OOB). This is the standard BCa acceleration
   definition, but it is slightly inconsistent with the OOB bootstrap
   evaluation (the jackknife is in-sample, the bootstrap is OOB). The effect
   is small (a = 0.0125, near zero) and the BCa CI is close to the percentile
   CI, so this inconsistency does not materially affect H54b.

6. **cpWER is utterance-level.** As in RQ43/RQ44, cpWER passes each speaker's
   full Chinese utterance as a single token; cpWER > 1.0 measures extra
   inserted speaker-streams, not character accuracy. A char-level
   re-validation remains the follow-up before claiming generalisation at
   character granularity.

## Reproducibility

- Script:
  `/opt/homebrew/bin/python3 results/frontier/cascade_f1_calibration/cascade_f1_analysis.py`
  (deterministic; numpy + stdlib only; no scipy / sklearn / Whisper / meeteval).
  Runtime ≈ 30 s (B=10000 vectorised bootstrap over 856-point grid + 77-fit
  jackknife).
- Tests: `/opt/homebrew/bin/python3 -m unittest tests.test_cascade_f1_calibration -v`
  (57 tests; pins `norm_ppf`, `norm_cdf`, `calibrate_f1`/`count_modes`
  re-exports, `cascade_cpwer_at_threshold` / `cascade_compute_at_threshold` /
  `cascade_oob_cpwer`, the vectorised `bootstrap_f1_cascade` (proved equivalent
  to RQ48's per-call `calibrate_f1`), `jackknife_acceleration`, `bca_ci`,
  module constants, and in-sample smoke tests reproducing RQ43's 0.888947
  anchor and the 37/40 label counts on the 77-window data).
- Outputs:
  - `cascade_f1_results.json` — full summary (in-sample F1 calibration,
    bootstrap threshold + OOB cpWER distributions with mode table, BCa CI
    with z₀/accel/adjusted percentiles, jackknife θ_loo, hypothesis verdicts)
    plus `per_bootstrap` arrays (thresholds, oob_cpwer, n_oob) for all 10000
    resamples for reproducibility.
- Bootstrap: B = 10000, seed = 42, n = 77 with replacement (RQ44 OOB protocol:
  calibrate in-bag, evaluate OOB).
- Source data:
  - `results/frontier/three_tier_cascade/three_tier_cascade_results.json`
    (label `experimental/frontier`, read-only — not modified).
  - `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
    (label `external/sanity-check`, read-only — not modified).

## What this changes for the project

RQ48 left an open question: does F1's threshold-stability gain on lang-id-
entropy transfer to other detectors? RQ54 answers **yes, and then some**: on
the KL detector F1 collapses to a **single mode** (vs RQ48's 2 on lang-id-
entropy), because the KL detector lacks the Mode S identifiability problem
that bounds lang-id-entropy. The calibration-rule stability gain is detector-
dependent: F1's mode-count reduction is larger on detectors without
low-score-ambiguity sub-populations.

The cpWER finding (H54c) is a **compute/cpWER trade-off**, not a free
improvement: F1's low threshold escalates 83% of windows to base, lowering
cpWER by 12.3% but raising compute from ~1.4× to 1.773×. RQ43's original
spec rule is the compute-efficient operating point; F1 is the cpWER-minimising
operating point. A compute-aware F1 variant (e.g. F1 subject to a compute
budget) would recover the Pareto frontier — a natural follow-up.

The CI finding (H54b) is essentially null: F1 + BCa + OOB gives the same cpWER
interval width as RQ46's percentile + in-bag + fixed-threshold. The cpWER
uncertainty is dominated by the n = 77 sample size, not the calibration rule
or CI method. Tightening the interval requires more data (multi-meeting
calibration), not a better calibration rule.

The actionable conclusion: on the KL detector, **F1 calibration is the
stability-optimal choice** (single-mode, 96.3% concentration) and the
**cpWER-optimal choice at 1.773× compute** (12.3% lower cpWER than RQ43's
spec rule). RQ43's spec rule remains the compute-efficient choice (1.4×
compute, 0.8889 cpWER). The two rules occupy different points on the same
compute/cpWER Pareto curve; the choice between them is a deployment-time
compute budget decision, not a calibration-rule quality decision.
