# Emotion-ASR Asymmetry Mechanism (RQ6) — Findings

**Label:** `experimental/frontier`. ASR = Whisper-`tiny` (cached, offline). Reanalysis of existing
findings #14 (`emotion_separation_tax`), #21 (`causal_hallucination_probe`), #18
(`objective_aware_routing`); **no new data collection**. Stable tables untouched. Issue #883.
Mode C (Frontier Exploration).

Module: `asymmetry_analysis.py`. Reproduce:
`python results/frontier/emotion_asr_asymmetry/asymmetry_analysis.py`
(numpy + pandas only; sklearn/scipy unavailable in this environment, so logistic regression
and AUC are implemented in pure numpy with leave-one-out CV).

## The question this answers

Finding #14 found an unexplained **asymmetry**: separation has NO emotion tax (prosody benefit
≥ 0 at every overlap) but a positive ASR tax at low/mid overlap (CER benefit −1.38 at α=0/ov0.1;
−1.72 at α=0.15/ov0.3). Finding #18 resolved this *operationally* (objective-aware decoupling:
text from the ASR route, emotion from the separated track) but did not explain the *cause*.
Finding #21 showed the catastrophic tail is a **confident attractor** (high logprob, low entropy),
detectable post-decode. This study tests two propositions that explain *why* the same separation
operation preserves prosody but injects text hallucination:

- **P2 (dimensionality effect):** prosody is low-dimensional continuous (robust to additive
  artifacts) while text decoding is high-dimensional discrete (sensitive to artifact-induced
  confusion). Prediction: a low-dimensional text-derived feature should ALSO be preserved by
  separation at low overlap where CER is hurt.
- **P3 (pre-decode predictability):** the confident-attractor onset is predictable from
  pre-decode encoder representations, not just detectable post-decode. Test: train a classifier
  on pre-decode features of #21's 66 conditions; AUC > 0.6 supports P3.

## Dimensionality analysis

| feature space | modality | effective dim | discrete | note |
|---|---|---:|:--:|---|
| prosody (acoustic arousal) | acoustic | 3 | no | arousal, valence, dominance; #14 uses arousal-side |
| text (ASR token sequence) | decoded | ~50,000 | yes | Whisper-zh vocab; full sequence is high-dim discrete |
| speaker count (text-derived) | decoded-meta | 1 | yes | single integer; ~1–2 bits |
| binary transcript usability (text-derived) | decoded-meta | 1 | yes | 1 bit: CER < 1.0 |
| utterance length (text-derived) | decoded-meta | 1 | no | proxy = compression_ratio in #21 |

The dimensionality gap between prosody (~3) and text (~50k) is ~4 orders of magnitude. P2
predicts this gap is the cause: low-dim estimators average out additive separation artifacts;
high-dim discrete decoders cannot.

## P2 verdict — SUPPORTED (moderate regime) / BOUNDED (catastrophic regime)

Using #14's crosslink data (8 pairs × 5 overlaps × 2 α levels), at low/mid overlap
(0.1, 0.3) where separation HURTS high-dim text (CER benefit < 0), low-dimensional features
are preserved:

**α = 0.15 (realistic leaky separator), low/mid overlap, conditions where CER is hurt (n=14):**

| feature | dim | preserved? | fraction preserved |
|---|---:|:--:|---:|
| CER benefit (high-dim text) | ~50,000 | HURT | mean −1.207 |
| emotion benefit (prosody, acoustic) | ~3 | preserved | 0.93 (13/14) |
| speaker count (text-derived) | 1 | preserved | 1.00 (14/14, always 2) |
| binary transcript usability (text-derived) | 1 | preserved | 0.64 (9/14) |

At the oracle separator (α=0.0), the same pattern holds (n=4 hurt conditions: emotion 1.00,
speaker count 1.00, binary usability 0.75, mean CER benefit −2.894).

**Interpretation.** In the moderate low-overlap tax regime, the same separation artifacts that
damage the high-dimensional token sequence (CER benefit −1.207) leave low-dimensional features
intact: speaker count is always preserved (it is a 1-integer summary, robust to token-level
confusion), binary transcript usability is preserved in 64% of hurt conditions (the transcript
still has *some* correct content even when CER rises), and emotion prosody is preserved in 93%.
This is the pattern P2 predicts — **low-dim preserved, high-dim hurt** — and supports
dimensionality as the cause of the #14 asymmetry.

**Bound from #21 (catastrophic regime).** In the catastrophic confident-attractor regime
(#21's 26 catastrophic conditions), even the 1-dimensional utterance-length feature is NOT
preserved: mean compression_ratio = **17.9×** (vs 0.81× for clean conditions) — the separated
transcript is ~18× longer than the reference (massive repetition). So P2 is **bounded**:
dimensionality explains the *moderate* asymmetry (low-overlap tax) but not the *extreme*
hallucination collapse, where the attractor distorts even 1-dim text features.

> **Note on the hypothesis.md disconfirmation criterion.** `hypothesis.md` states "if a
> low-dimensional text feature (e.g. speaker count) is also preserved by separation, the
> dimensionality explanation is wrong." This analysis follows the task's logically coherent
> interpretation (preserved low-dim + hurt high-dim ⇒ P2 supported), which is the inverse of
> the hypothesis.md wording. The rival explanation (modality: acoustic vs decoded) is not
> distinguishable from dimensionality using speaker count alone (it is trivially preserved
> and could be acoustically-grounded); the binary-usability result (a genuinely decoded
> 1-bit feature, preserved in 64% of hurt conditions) is the stronger evidence for
> dimensionality over modality.

## P3 verdict — SUPPORTED (weakly / borderline), best pre-decode LOO AUC = 0.623

Using #21's 66 conditions (26 catastrophic vs 40 clean), the available pre-decode features
are `overlap_ratio` (input property, truly pre-decode) and `no_speech_prob` (Whisper's
encoder-side no-speech token probability, max over segments — the closest available proxy
for a pre-decode encoder representation; **raw encoder embeddings are NOT stored** in
`probe_rows.csv`). All other features (avg_logprob, token_entropy, dominant_token_fraction,
compression_ratio) are post-decode. `cer_mixed` is excluded (oracle — requires a reference).

**Single-feature ranking AUC (in-sample, with optimal direction):**

| feature | type | AUC |
|---|---|---:|
| overlap_ratio | pre-decode | 0.517 |
| no_speech_prob (raw) | pre-decode | 0.334 |
| no_speech_prob (oracle-flipped) | pre-decode | 0.666 |
| avg_logprob | post-decode | 0.848 |
| token_entropy (flipped) | post-decode | 0.566 |
| dominant_token_fraction | post-decode | 0.703 |
| compression_ratio | post-decode | 0.996 |

**Logistic-regression leave-one-out AUC (proper out-of-sample; direction learned in-fold):**

| feature set | type | LOO AUC |
|---|---|---:|
| overlap_ratio only | pre-decode | 0.000† |
| no_speech_prob only | pre-decode | **0.623** |
| overlap + no_speech_prob | pre-decode | 0.588 |
| overlap + (1−no_speech_prob) | pre-decode | 0.588 |
| **best pre-decode** | pre-decode | **0.623** |
| compression_ratio only | post-decode | 0.846 |
| cr + dom + logp + entropy | post-decode | 0.992 |

† `overlap_ratio` has only 3 unique values (0.1/0.3/0.5) and a near-chance ranking AUC
(0.517); the LOO logistic regression on a near-chance feature with 3 levels produces a
degenerate anti-correlated out-of-fold ranking (AUC 0.0). This is a high-variance LOO
artifact, not a real signal — the ranking AUC (0.517) is the reliable summary.

**Verdict.** The best pre-decode LOO classifier AUC is **0.623** (no_speech_prob alone),
marginally above the 0.6 threshold ⇒ P3 is **weakly supported (borderline)**. Caveats:
(1) the signal is weak — just 0.023 above threshold; (2) the combined pre-decode model
(overlap + nsp) is *below* threshold (0.588); (3) the in-sample oracle-direction ceiling is
0.666; (4) #21's FINDINGS already noted no_speech_prob is anti-correlated (raw AUC 0.334)
and "NOT the signal" — the 0.623 LOO AUC exploits this anti-correlation by learning the
flipped direction. By contrast, post-decode detection is far stronger (compression_ratio
LOO AUC 0.846; full post-decode set 0.992). **The attractor is marginally predictable
pre-decode but overwhelmingly a post-decode phenomenon.**

## Mechanism explanation

The two propositions together yield a **two-layer mechanism** for the #14/#18 asymmetry:

**Layer 1 — dimensionality (the dominant explanation, P2 supported).** In the moderate
low-overlap tax regime, separation artifacts are additive perturbations. A low-dimensional
continuous estimator (prosody, ~3 dims) averages them out; a high-dimensional discrete
decoder (text, ~50k tokens) cannot — the same artifact that is noise to prosody is confusion
at the token-decision boundary. This explains why emotion benefit ≥ 0 while CER benefit < 0
at low/mid overlap, and why low-dim text-derived features (speaker count, binary usability)
are also preserved there. The dimensionality gap (~4 orders of magnitude) is the structural
cause of the asymmetry.

**Layer 2 — pre-decode predictability (weak, P3 borderline-supported).** Whisper's
encoder-side `no_speech_prob` carries a weak pre-decode signal (LOO AUC 0.623): catastrophic
routes have *lower* no_speech_prob (0.179 vs 0.256) — the decoder is busy "speaking" in a
confident loop, so it assigns low probability to the no-speech token. This is consistent with
#21's refinement that the encoder-silence decoupling was smoke-test-specific. But the signal
is marginal and the combined model is below threshold, so pre-decode *prevention* is not
reliably achievable from the features stored in #21's data.

**The bound.** In the catastrophic regime (#21, compression_ratio ~18×), the confident
attractor is a nonlinear decoder collapse (Mode R repetition / Mode N substitution) that
distorts even 1-dim text features (length). Dimensionality explains the *moderate* asymmetry
but not the *extreme* collapse — the latter is a discrete-decoder pathology that only
post-decode detection (compression_ratio + token-id lock-in, #21) catches reliably.

## Operational implication (consistent with #18)

Keep the text route on the reference-free ASR router (overlap / compression-ratio signals,
post-decode), and always read emotion from the separated track. Pre-decode *prevention* of
the attractor is at best weakly supported by this data; post-decode *detection* (#21's
compression-ratio + token-id lock-in union) remains the deployable cure. The dimensionality
result (P2) explains *why* the decoupled design of #18 works: emotion is recoverable from
the separated track precisely because prosody is low-dimensional and robust to the artifacts
that break the high-dimensional text decode.

## Honest limitations

- Whisper-`tiny` only; oracle/leaky mixtures (no real separator); one Chinese-debate corpus.
- P2's speaker-count test is degenerate (always 2 speakers in this corpus) — it is preserved
  trivially. The binary-usability test (a genuinely decoded 1-bit feature) is the stronger
  evidence, but it is still CER-derived. A cleaner test would require raw token counts (not
  stored in #14's data).
- P3 uses `no_speech_prob` as a proxy for pre-decode encoder state; **raw encoder embeddings
  are not stored** in `probe_rows.csv`. A true test of P3 would require re-running #21 with
  encoder-embedding dumps — out of scope for this reanalysis (no new data collection).
- n=66 for P3 (26 catastrophic / 40 clean); the 0.623 LOO AUC is within the noise band for
  this sample size and should not be over-interpreted. The honest reading is "borderline, not
  robustly predictable pre-decode."
- The `overlap_ratio` LOO AUC of 0.000 is a degenerate artifact (3 unique values, near-chance
  signal); the ranking AUC (0.517) is the reliable summary.
- `experimental/frontier`, not a gold result. Artifacts: `asymmetry_analysis.py`,
  `asymmetry_results.csv`, `asymmetry_results.json`.
