# RQ38: Speaker-Count Effect on Hallucination Rate (AISHELL-4)

> **Label: `experimental/frontier`** — reanalysis-only test of whether the number of
> active speakers per window correlates with separated-track hallucination rate on
> AISHELL-4, and whether speaker count predicts Mode S specifically. No Whisper / no
> ASR model is run; this reads the existing AISHELL-4 external-validation results and
> computes correlations, stratified rates, and partial correlations. Closes #945.
>
> Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
> (label `external/sanity-check`, PR #890). Prior context: RQ12
> (`results/frontier/router_failure_modes/FINDINGS.md`, "silence gaps drive
> hallucination, not multi-speaker confusion" — only 2 of 11 failure windows had
> >2 speakers), RQ19 (`results/frontier/mode_s_detector/FINDINGS.md`, the 2 Mode S
> residual windows 22 and 30), and RQ37 (per-speaker cpWER decomposition; the task
> brief reports speaker 001-M dominates the AISHELL-4 reference — its FINDINGS file
> was not present at analysis time, so RQ37 is cited via the brief).

## Executive Summary

Across the 77 AISHELL-4 windows, the number of active speakers per window (speakers
with a non-empty separated transcript) is **strongly and monotonically** related to
the separated-track hallucination rate. The relationship is strikingly clean:

| Active speakers | n | Hallucination rate | Bootstrap 95% CI |
|---|--:|---:|---|
| 0 | 13 | 0.0% | [0.0%, 0.0%] |
| 1 | 29 | 34.5% | [17.2%, 51.7%] |
| 2 | 20 | 65.0% | [45.0%, 85.0%] |
| 3+ | 15 | 93.3% | [80.0%, 100.0%] |

Spearman ρ between active speaker count and the binary hallucination label is
**+0.611** (permutation p < 0.0001, 10,000 perms, seed=42); against the continuous
separated cpWER it is **+0.536** (p < 0.0001). The configured `num_speakers` field
gives ρ = +0.473. All three clear the H38a bar of ρ > 0.2 by a wide margin.

**H38a (positive correlation): SUPPORTED. H38b (Mode S is low-speaker-count):
SUPPORTED. H38c (mediation by silence fraction): NOT SUPPORTED.**

The H38c result is the scientifically interesting one. The pre-registered mediation
hypothesis — that the speaker-count effect is carried by silence-gap fraction — is
**rejected by both silence proxies available at the transcript level**:

- Partial correlation of active speakers with hallucination, **controlling for the
  per-speaker silence fraction** (fraction of present speakers with an empty
  separated transcript) = **+0.538**. This is essentially unchanged from the raw
  +0.611 — the silence fraction removes almost none of the speaker-count effect.
- Controlling for **length_ratio** (separated length / mixed length, a second
  silence-gap proxy) = **+0.618** — if anything slightly *larger*.
- In the **active ≥ 1 subsample** (which removes the 13 all-empty-decode windows
  that score cpWER = 1.0 exactly and are mechanically non-hallucinated), the simple
  ρ is still **+0.478** (p < 0.0001), and the partial correlations stay at +0.48.

So the speaker-count effect is **robust**: it survives removing the structural
floor and survives controlling for both transcript-level silence proxies. The
mediation hypothesis fails not because speaker count is irrelevant, but because the
per-speaker empty fraction is the *wrong* silence measure — it captures "Whisper
produced no output" (which yields cpWER = 1.0, i.e. *non*-hallucination), the
opposite of RQ12's audio-level silence-gap stimulus. The speaker count almost
certainly proxies for the audio-level inter-speaker silence gaps RQ12 identified,
but that stimulus is not recoverable from the stored transcripts.

## Hypothesis Verdicts

### H38a — Positive correlation between active speaker count and hallucination rate (ρ > 0.2): **SUPPORTED**

- ρ(active speakers, hallucination) = **+0.6107**, permutation p < 0.0001.
- ρ(active speakers, separated cpWER) = **+0.5358**, permutation p < 0.0001.
- ρ(num_speakers, hallucination) = **+0.4733**, permutation p < 0.0001 (secondary
  measure, using the configured TextGrid speaker count rather than the Whisper-decoded
  active count).
- Monotonic stratification: 0.0% → 34.5% → 65.0% → 93.3% across active {0,1,2,3+}.

All three correlations clear the ρ > 0.2 bar by a factor of 2-3×, and the
stratification is monotone with non-overlapping CIs between the extreme strata
(0% [0,0] vs 93.3% [80%,100%]). The effect is not subtle.

**Caveat (load-bearing):** the active=0 stratum (13 windows) is **structurally**
non-hallucinated — when Whisper produces no output for any speaker, the separated
cpWER is exactly 1.0 (pure deletions, no insertions), which the `> 1.0` label marks
as non-hallucinated. This inflates the positive correlation. However, the effect
persists after removing this floor: in the active ≥ 1 subsample (n=64), ρ = +0.478
(p < 0.0001) and the stratified rates still climb 34.5% → 65.0% → 93.3%. So the
correlation is not an artefact of the structural floor alone.

### H38b — Mode S windows (w22, w30) have ≤ 2 active speakers: **SUPPORTED**

| Window | num_speakers | active_speakers | overlap | length_ratio | cpWER |
|---|---:|---:|---|---:|---:|
| w22 | 2 | 1 | NoOverlap | 1.021 | 2.000 |
| w30 | 1 | 1 | NoOverlap | 1.027 | 2.000 |

Both Mode S windows have ≤ 2 active speakers (and ≤ 2 configured speakers). This is
consistent with RQ37's report (via the task brief) that speaker 001-M dominates the
AISHELL-4 reference, and with RQ19's finding that Mode S is a monoscript-Chinese
near-duplicate of the mixed decode produced when the separator fails on a single
dominant speaker's audio. Mode S is a **low-speaker-count, low-overlap** phenomenon
— the opposite end of the speaker-count axis from the diverse hallucination that
dominates the 3+ stratum. The length_ratio values (~1.02) reproduce RQ19's Mode S
signature exactly, cross-validating the Mode S labels.

### H38c — Speaker-count effect mediated by silence fraction (partial ρ < 0.1): **NOT SUPPORTED**

| Partial correlation of (active speakers, hallucination) controlling for… | Value |
|---|---:|
| silence_fraction (per-speaker empty fraction), all windows | **+0.538** |
| length_ratio (sep/mixed length), all windows | +0.618 |
| silence_fraction, active ≥ 1 subsample | +0.481 |
| length_ratio, active ≥ 1 subsample | +0.492 |

None of the four partial correlations drops below the 0.1 bar. The speaker-count
effect is **not mediated** by either transcript-level silence proxy. Two
observations make this interpretable rather than just a null:

1. **The per-speaker silence fraction is the wrong measure of RQ12's stimulus.**
   silence_fraction vs hallucination has ρ = **−0.348** (p = 0.0022) — it is
   *negatively* related to hallucination, because high silence_fraction (many empty
   speakers) coincides with all-empty decodes that score cpWER = 1.0 and are
   non-hallucinated. RQ12's silence-gap stimulus is an **audio-level interior-silence**
   property that triggers Whisper *insertions* (cpWER > 1.0); the per-speaker empty
   fraction captures the opposite event (Whisper *produces nothing*). They are
   different failure modes, and one cannot mediate the other.

2. **length_ratio is a weaker but better-aligned proxy, and it still does not
   mediate.** ρ(length_ratio, hallucination) = +0.094 (p = 0.41, n.s.) — length_ratio
   is nearly independent of hallucination on this data. Controlling for it leaves the
   speaker-count effect essentially intact (+0.618). So even the more independent
   transcript-level silence proxy does not carry the speaker-count effect.

The honest reading: the speaker-count effect is real and robust, but the
transcript-level silence proxies available in this dataset **cannot** capture the
audio-level silence-gap mechanism RQ12 identified. H38c's failure is therefore a
**measurement limitation**, not a refutation of RQ12's mechanism. Resolving whether
the speaker-count effect *is* the silence-gap effect (just unmeasurable here)
requires either audio-level silence features (RQ8's silence-aware gate) or a
controlled dataset where speaker count and silence-gaps are decorrelated.

## Method

### Data source (read-only, not overwritten)

`results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
(label `external/sanity-check`, PR #890): 77 windows × 30 s from AISHELL-4 meeting
M_R003S02C01 (6 speakers, oracle-TextGrid separation, Whisper-tiny, MeetEval
cpWER/orcWER). Each window stores `num_speakers` (speakers present per TextGrid),
`separated_text_per_speaker` (one entry per present speaker), `separated_total_length`,
`mixed_text_length`, `runtime_ratio`, and the cpWER each route would yield.

### Per-window features (pure, deterministic)

- `active_speakers` = count of non-empty `separated_text_per_speaker` values
  (whitespace-stripped). This is the Whisper-decoded speaker count, which can be
  less than `num_speakers` when Whisper produces no output for a present speaker.
- `num_speakers` = the configured TextGrid speaker count (the JSON field).
- `silence_fraction` = (num_speakers − active_speakers) / num_speakers — fraction of
  present speakers with an empty separated transcript. A transcript-level silence
  proxy (H38c's primary control).
- `length_ratio` = separated_total_length / mixed_text_length — a second silence-gap
  proxy, more independent of speaker count (H38c's robustness control).
- `hallucinated` = `always_separated_cpwer > 1.0` (RQ12's split: 37/77). Strict `>`
  so that the all-empty-decode cpWER of exactly 1.0 is non-hallucinated.
- `mode_s` = hallucinated AND window_id ∈ {22, 30} (RQ19's definition, reproduced
  exactly: both windows have length_ratio ≈ 1.02, NoOverlap).

### Statistics

- **Spearman ρ** with **10,000-permutation** two-sided tests (seed=42, +1 smoothing)
  for: active vs cpWER, active vs hallucination, num_speakers vs hallucination,
  silence_fraction vs hallucination, length_ratio vs hallucination.
- **Stratified hallucination rates** for active {0,1,2,3+} and num_speakers
  {1,2,3,4,5+}, each with **bootstrap 95% CIs** (10,000 resamples, seed=42).
- **Rank-based partial correlations** (rank-transform all three variables, residualise
  the control out of x and y by OLS, Pearson-correlate the residuals) of active vs
  hallucination, controlling for silence_fraction and length_ratio, both on all 77
  windows and on the active ≥ 1 subsample (which removes the structural cpWER=1.0
  floor).
- **Active ≥ 1 subsample** (n=64) simple ρ as a robustness check on H38a, isolating
  whether the correlation survives once the all-empty-decode floor is removed.

All code is numpy + scipy + stdlib only (no sklearn, no Whisper, no audio). Pure
helpers are unit-tested in `tests/test_speaker_count_effect.py` (51 tests).

## Results

### H38a — correlations and stratification

| Pair | Spearman ρ | Permutation p (10k) |
|---|---:|---:|
| active speakers × hallucination | **+0.611** | < 0.0001 |
| active speakers × separated cpWER | **+0.536** | < 0.0001 |
| num_speakers × hallucination | +0.473 | < 0.0001 |
| silence_fraction × hallucination | −0.348 | 0.0022 |
| length_ratio × hallucination | +0.094 | 0.4148 |

Hallucination rate by active-speaker stratum (bootstrap 95% CI):

| Stratum | n | Hallucination rate | 95% CI |
|---|--:|---:|---|
| 0 | 13 | 0.000 | [0.000, 0.000] |
| 1 | 29 | 0.345 | [0.172, 0.517] |
| 2 | 20 | 0.650 | [0.450, 0.850] |
| 3+ | 15 | 0.933 | [0.800, 1.000] |

Hallucination rate by configured num_speakers (bootstrap 95% CI):

| num_speakers | n | Hallucination rate | 95% CI |
|---|--:|---:|---|
| 1 | 22 | 0.227 | [0.091, 0.409] |
| 2 | 32 | 0.406 | [0.250, 0.594] |
| 3 | 16 | 0.750 | [0.500, 0.938] |
| 4 | 6 | 1.000 | [1.000, 1.000] |
| 5+ (n=1, 6-speaker window) | 1 | 1.000 | [1.000, 1.000] |

Both stratifications are monotone increasing. The 13 active=0 windows are all
non-hallucinated at cpWER = 1.0 exactly (the structural floor). The active ≥ 1
subsample (n=64) still hallucinates at 57.8% (37/64) with ρ = +0.478 (p < 0.0001),
so the effect is not driven solely by the floor.

### H38b — Mode S speaker counts

| Window | num_speakers | active_speakers | overlap | length_ratio | cpWER |
|---|---:|---:|---|---:|---:|
| w22 | 2 | 1 | NoOverlap | 1.021 | 2.000 |
| w30 | 1 | 1 | NoOverlap | 1.027 | 2.000 |

Both Mode S windows have ≤ 2 active speakers. The length_ratio values (~1.02)
reproduce RQ19's Mode S signature, cross-validating the labels. Both are
NoOverlap — Mode S sits at the low-speaker-count, low-overlap end of the axis.

### H38c — partial correlations (mediation test)

| Control variable | Sample | Partial ρ (active × halluc) |
|---|---|---:|
| silence_fraction | all 77 | **+0.538** |
| length_ratio | all 77 | +0.618 |
| silence_fraction | active ≥ 1 (n=64) | +0.481 |
| length_ratio | active ≥ 1 (n=64) | +0.492 |

Reference: simple ρ(active × halluc) = +0.611 (all) / +0.478 (active ≥ 1).

None of the partial correlations drops below 0.1. The speaker-count effect is not
mediated by either transcript-level silence proxy. The per-speaker silence fraction
is itself *negatively* correlated with hallucination (ρ = −0.348) because it captures
the all-empty-decode floor rather than RQ12's insertion-driving audio silence gaps.

## Honest Limitations

1. **Single AISHELL-4 meeting (n=77, 1 meeting).** Only M_R003S02C01 is available.
   The 6-speaker distribution is sparse at the high end (1 window with 6 speakers, 2
   with 4, 12 with 3), so the 3+ stratum's 93.3% rate rests on 15 windows. The
   monotone trend is clear, but the exact stratum rates are indicative, not precise.

2. **Structural floor at active=0.** 13 windows have Whisper produce no output for any
   speaker, scoring cpWER = 1.0 (pure deletions) and being mechanically
   non-hallucinated by the `> 1.0` label. This inflates H38a's ρ. We report the
   active ≥ 1 subsample (ρ = +0.478) to bound this, but the floor is inherent to the
   `cpWER > 1.0` hallucination operationalisation and cannot be fully removed without
   redefining the label.

3. **The silence proxies are transcript-level, not audio-level.** RQ12's silence-gap
   mechanism is an audio-level interior-silence property that triggers Whisper
   insertions. Neither the per-speaker empty fraction nor length_ratio captures it:
   the empty fraction captures the *opposite* event (Whisper produces nothing), and
   length_ratio is nearly uncorrelated with hallucination (ρ = +0.094). H38c's
   failure therefore reflects a measurement gap, not a refutation of the silence-gap
   mechanism. A clean test of H38c would need audio-level silence features (RQ8's
   silence-aware gate) or a dataset where speaker count and silence gaps are
   decorrelated.

4. **Active speaker count and num_speakers are distinct but correlated.** Active
   count (Whisper-decoded) can be less than num_speakers (TextGrid-present) when
   Whisper drops a speaker. The two measures give ρ = +0.611 and +0.473 respectively
   — same direction, slightly different magnitude. We report both; the active count
   is the primary measure because it reflects what Whisper actually decoded.

5. **Active speaker count is partly downstream of hallucination.** When Whisper
   hallucinates a long multi-script track it tends to fill multiple speaker slots
   (window 0: 6 active, all non-empty, cpWER 2.33). So active count is not purely an
   exogenous predictor; it is partly a consequence of Whisper's decoding behaviour.
   The correlation is real but the causal direction (more speakers → more
   hallucination vs more hallucination → more active speakers) cannot be settled from
   transcripts alone.

6. **Oracle-TextGrid separation.** The separated tracks use oracle boundaries (true
   silence gaps), not a real separator (residual noise). The speaker-count effect is
   specific to this separation paradigm; a real separator may produce a different
   speaker-count / hallucination relationship.

7. **RQ37 citation is second-hand.** The task brief reports RQ37 (per-speaker cpWER
   decomposition) found speaker 001-M dominates; RQ37's FINDINGS file was not present
   at analysis time (`results/frontier/per_speaker_cpwer_decomposition/FINDINGS.md`
   does not exist in this worktree). H38b's consistency with the 001-M-dominance
   claim is therefore stated as directional agreement, not a direct reproduction.

## What this changes for the project

1. **Speaker count is a strong, monotone predictor of separated-track hallucination
   on AISHELL-4** (ρ ≈ +0.61). This refines RQ12's "multi-speaker is not the driver"
   finding: RQ12 looked at the 11 *router-failure* windows and found only 2 had >2
   speakers; RQ38 looks at the full 77-window hallucination *rate* and finds a clean
   monotone increase. Both can be true — router failures concentrate at low speaker
   counts (Mode S), while the *base rate* of hallucination climbs with speaker count.
   Speaker count is a useful risk feature for hallucination, not because of
   multi-speaker *confusion* (RQ12 ruled that out), but because more speakers means
   more inter-speaker silence gaps (RQ12's mechanism).

2. **Mode S is confirmed as a low-speaker-count, low-overlap phenomenon** (H38b),
   consistent with RQ37's 001-M dominance and RQ19's near-duplicate mechanism. Mode S
   and the diverse hallucination that dominates the 3+ stratum sit at **opposite
   ends** of the speaker-count axis — they are different failure modes, and a
   hallucination detector should not treat them as one.

3. **The transcript-level silence proxies do not mediate the speaker-count effect
   (H38c failed).** This is a negative result with a positive implication: to test
   whether the speaker-count effect *is* the silence-gap effect, the project needs
   audio-level silence features (RQ8's silence-aware gate), not transcript-level
   length or empty-fraction proxies. This is a concrete pointer for the next step:
   wire RQ8's audio silence-gap fraction into this analysis and re-test H38c. If the
   audio-level silence fraction mediates the speaker-count effect (partial ρ < 0.1),
   the RQ12 mechanism is confirmed at the correlation level; if it does not, the
   speaker-count effect is a distinct risk factor.

4. **The structural floor (active=0 → cpWER=1.0) is a measurement caveat for all
   cpWER-based hallucination analyses on this dataset.** 13 of 77 windows sit at it.
   Future hallucination-rate analyses should report both the all-windows and the
   active ≥ 1 rates to avoid over-stating effects that lean on the floor.

## Reproducibility

- Script: `python3 results/frontier/speaker_count_effect/speaker_count_effect_analysis.py`
  (deterministic; numpy + scipy + stdlib only; no sklearn / Whisper / audio). Runs in ~15 s.
- Tests: `python3 -m unittest tests.test_speaker_count_effect` (51 tests on pure helpers).
- Outputs:
  - `speaker_count_effect_results.csv` (per-window: window_id, num_speakers,
    active_speakers, empty_speakers, silence_fraction, length_ratio, runtime_ratio,
    always_separated_cpwer, always_mixed_cpwer, hallucinated, mode_s, overlap_label).
  - `speaker_count_effect_results.json` (label, distributions, correlations +
    permutation p-values, stratification tables with bootstrap CIs, partial
    correlations, hypothesis verdicts).
- Bootstrap: 10,000 resamples, seed=42. Permutation: 10,000 permutations, seed=42.
- Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
  (label `external/sanity-check`, read-only — not modified). No verified references
  or gold tables were modified.
