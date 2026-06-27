# RQ60: KL+lang-id Ensemble Corrected Router

> **Label: `experimental/frontier`** — a reanalysis-only simulation that tests
> OR and AND ensembles of two complementary hallucination detectors (KL
> divergence from RQ58 and lang-id entropy from RQ13/RQ16) in the corrected
> router. Does NOT run Whisper, any ASR model, or any LLM/ollama. Does NOT
> overwrite any verified reference or gold table. Closes #984.
>
> Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
> (label `external/sanity-check`, PR #890, read-only). KL detector primitives
> are imported VERBATIM from `src/llm_semantic_critic.py` (RQ34, PR #951); BCa
> CI helpers are reimplemented from RQ39
> (`results/frontier/bootstrap_ci_corrected_router/bootstrap_ci_analysis.py`,
> PR #955). RQ58 reference values are from
> `results/frontier/kl_corrected_router/` (PR #981). RQ16 reference values are
> from `results/frontier/corrected_router_simulation/` (PR #909).

## Executive Summary

RQ58 (PR #981) showed the n-gram KL detector catches Mode S (0%→100%) but has 4
false positives, achieving cpWER **1.030**. RQ16 (PR #912) showed lang-id
entropy catches 94.6% of all hallucinations but 0% of Mode S, achieving cpWER
**1.043**. The two detectors are COMPLEMENTARY in detection: KL catches Mode S
(which lang-id misses), lang-id catches diverse hallucinations (which KL also
catches). RQ60 asks: does an OR/AND ensemble of KL and lang-id beat both
individually?

**No.** The OR ensemble achieves cpWER **1.030** — identical to KL-alone — and
the AND ensemble achieves cpWER **1.043** — identical to lang-id-alone. Neither
ensemble improves on the better individual detector. The reason is structural:
KL-alone already catches **100%** of all 37 hallucinations (including both Mode
S windows), so the OR ensemble cannot add any true positives — it can only add
false positives. The 2 additional OR false positives (windows 19 and 44, where
lang-id flags but KL does not) happen to fall on cpWER-tied windows
(`always_mixed_cpwer == always_separated_cpwer == 1.0`), so they are
cpWER-neutral: routing them to MIXED instead of SEPARATED changes nothing. The
AND ensemble reduces false positives (from 3–4 to 1) but misses Mode S (since
Mode S has low lang-id entropy), so its cpWER equals lang-id-alone.

| Policy | cpWER | vs KL-alone | vs lang-id | FP | Mode S sens |
|--------|------:|------:|------:|---:|---:|
| always-mixed | 1.1732 | — | — | 40 | 100% |
| always-separated | 1.5909 | — | — | 0 | 0% |
| router v2 | 1.2056 | — | — | — | — |
| oracle best | 1.0173 | — | — | — | 100% |
| RQ16 lang-id-alone | 1.0433 | +0.013 | — | 3 | 0% |
| RQ58 KL-alone | 1.0303 | — | −0.013 | 4 | 100% |
| **OR ensemble (RQ60)** | **1.0303** | **0.000** | **−0.013** | **6** | **100%** |
| **AND ensemble (RQ60)** | **1.0433** | **+0.013** | **0.000** | **1** | **0%** |

**Hypothesis verdicts:**

| Hypothesis | Verdict | Detail |
|---|:---:|---|
| **H60a** OR cpWER < 1.030 (beats KL-alone) | **KILLED** | OR 1.030303 >= 1.030; Δ = +0.000000 (identical to KL-alone) |
| **H60b** OR catches 100% Mode S AND >= 90% all | **SUPPORTED** | OR catches 100% Mode S (2/2) and 100% all (37/37) |
| **H60c** AND FP rate < 7.5% (3/40) | **SUPPORTED** | AND FP = 1/40 = 2.5% < 7.5%; KL has 4 FPs, lang-id has 3 |

The H60a kill is the central finding: **the OR ensemble cannot beat KL-alone
because KL-alone is already at 100% sensitivity**. The ensemble's additional
lang-id flags add false positives without adding true positives, and those
false positives happen to be cpWER-neutral (mixed == separated). The ensemble
is structurally redundant when one detector already catches everything.

The H60c support is a secondary finding: the AND ensemble achieves the lowest
false-positive rate of any detector tested (1/40 = 2.5%, vs KL's 4/40 and
lang-id's 3/40), but at the cost of missing Mode S — so its cpWER (1.043)
matches lang-id-alone, not KL-alone.

## Method

### Data

77 windows of 30 s from AISHELL-4 meeting `M_R003S02C01` (6 speakers, 38.5 min).
Each window stores the per-route cpWER (`always_mixed_cpwer`,
`always_separated_cpwer`, `router_v2_cpwer`, `oracle_best_cpwer`) and the
per-speaker separated transcripts. No ASR is run; each ensemble's per-window
cpWER is the chosen route's stored word-level cpWER — matching RQ16/RQ58
bit-for-bit.

### Detectors

**KL detector (RQ58, pure statistical — NO LLM).** Character 2-gram
KL-divergence anomaly detector. The reference 2-gram distribution is the
average of the 40 non-hallucinated tracks' concatenated separated text (RQ34's
`build_reference_distribution`, n=2, vocab 1487). The per-window score is the
MAX-across-speakers KL (RQ12/RQ13 worst-case-track convention). Threshold
**5.418144** (RQ58's empirically-calibrated threshold at >=90% specificity; the
task spec says "5.42" rounded). KL-alone catches 100% of all hallucinations
including Mode S, with 4 false positives.

**lang-id entropy (RQ13/RQ16).** Shannon entropy over Unicode script
categories (Han, Latin, Hiragana, etc.), MAX-across-speakers. Clean Chinese is
near-monoscript (entropy ~ 0); diverse multilingual gibberish has high entropy.
Threshold **0.38** (task spec pre-registered). NOTE: threshold 0.38 gives
IDENTICAL results to RQ16's 0.409 — no lang-id score falls between 0.38 and
0.409 — so both reproduce RQ16's 1.043290 cpWER with 3/40 FPs and 94.6%
sensitivity. lang-id catches 35/37 hallucinations (94.6%) but 0% of Mode S.

### Ensemble rules

- **OR ensemble**: route to MIXED if `KL >= 5.418144 OR lang_id >= 0.38`. The
  OR ensemble is a superset of each detector's flags — it catches everything
  either detector catches. Maximises sensitivity at the cost of more FPs.
- **AND ensemble**: route to MIXED if `KL >= 5.418144 AND lang_id >= 0.38`. The
  AND ensemble is the intersection — it flags only windows where both detectors
  agree. Minimises FPs at the cost of sensitivity (misses windows where only
  one detector fires — e.g. Mode S, which has high KL but low lang-id).

### Statistics

Per-window cpWER is averaged over the 77 windows. Bootstrap 95% CIs use 10,000
resamples (seed=42) with the RQ39 framework: percentile CI, BCa CI
(bias-corrected + accelerated, with jackknife acceleration), and paired-delta
CI (per-window ensemble minus comparator, paired design). The BCa CI is the
primary CI. The bootstrap convention is `rng.integers(0, n, size=n)` per
resample (RQ16/RQ39 verbatim).

### Reproducibility sanity checks

The RQ58 KL-alone router and RQ16 lang-id-alone router are recomputed in-script
as sanity checks:
- `kl_alone_cpwer_recomputed = 1.030303` matches RQ58's `kl_corrected_router_cpwer`
  (PR #981) bit-for-bit. ✓
- `lang_id_alone_cpwer_recomputed = 1.043290` matches RQ16's
  `corrected_router_cpwer` (PR #909) bit-for-bit. ✓

The baselines (always-mixed 1.17316, always-separated 1.590909, router-v2
1.205628, oracle 1.017316) reproduce the source JSON. This confirms the cpWER
pipeline matches RQ16/RQ39/RQ58 before the new ensemble routing is applied.

## Results

### Detector evaluation

| Detector | Sensitivity (all 37) | Sensitivity (Mode S) | Specificity | FP | TP |
|---|---:|---:|---:|---:|---:|
| KL-alone (RQ58) | 100.0% (37/37) | 100% (2/2) | 90.0% | 4 | 37 |
| lang-id-alone (RQ16) | 94.6% (35/37) | 0% (0/2) | 92.5% | 3 | 35 |
| **OR ensemble** | **100.0% (37/37)** | **100% (2/2)** | **85.0%** | **6** | **37** |
| **AND ensemble** | **94.6% (35/37)** | **0% (0/2)** | **97.5%** | **1** | **35** |

The OR ensemble catches everything KL catches (100% all, 100% Mode S) plus 2
additional false positives (windows 19 and 44, where lang-id flags but KL does
not). The AND ensemble catches everything lang-id catches (94.6% all, 0% Mode
S) minus 2 false positives (the same windows 19 and 44, where lang-id flags but
KL does not — AND removes them), leaving only 1 FP (window 52, where both
detectors agree on a false positive).

### Aggregate cpWER (mean over 77 windows)

| Policy | cpWER | percentile CI 95% | BCa CI 95% |
|--------|------:|:---:|:---:|
| always-mixed | 1.1732 | — | — |
| always-separated | 1.5909 | — | — |
| router v2 | 1.2056 | — | — |
| oracle best | 1.0173 | — | — |
| RQ16 lang-id-alone | 1.0433 | [1.0087, 1.0887] | [1.0130, 1.0974] |
| RQ58 KL-alone | 1.0303 | [1.0043, 1.0671] | [1.0065, 1.0779] |
| **OR ensemble** | **1.0303** | **[1.0043, 1.0671]** | **[1.0065, 1.0779]** |
| **AND ensemble** | **1.0433** | **[1.0087, 1.0887]** | **[1.0130, 1.0974]** |

The OR ensemble's point estimate (1.030303), percentile CI, and BCa CI are
**bit-for-bit identical** to KL-alone. The AND ensemble's are **bit-for-bit
identical** to lang-id-alone. This is not a coincidence — it is the structural
consequence of where the two detectors disagree (see below).

### Why OR == KL-alone and AND == lang-id-alone (exactly)

The two detectors disagree on exactly **2 windows** (19 and 44), both
non-hallucinated. On these windows:
- lang-id flags (entropy > 0.38) but KL does not (score < 5.418144).
- `always_mixed_cpwer == always_separated_cpwer == 1.0` (cpWER tie).

| Window | hallucinated | KL flag | lang-id flag | mixed cpWER | sep cpWER | OR decision | AND decision |
|---|:---:|:---:|:---:|---:|---:|:---:|:---:|
| 19 | No | 0 | 1 | 1.0 | 1.0 | mixed | separated |
| 44 | No | 0 | 1 | 1.0 | 1.0 | mixed | separated |

Since these are the ONLY windows where OR differs from KL (and AND differs from
lang-id), and both have `mixed == separated == 1.0`, the routing choice is
cpWER-neutral:
- **OR** routes them to MIXED (additional FPs), but cpWER is unchanged → OR
  cpWER == KL cpWER.
- **AND** routes them to SEPARATED (removes lang-id FPs), but cpWER is unchanged
  → AND cpWER == lang-id cpWER.

On all other 75 windows, OR agrees with KL and AND agrees with lang-id (both
detectors either flag or don't flag together), so the ensembles' decisions
match the individual detectors.

### Per-window Mode S detail

| Window | KL score | KL flag | lang-id entropy | lang-id flag | OR decision | AND decision | OR cpWER | AND cpWER |
|---|---:|:---:|---:|:---:|:---:|:---:|---:|---:|
| 22 | 13.079 | ✓ | 0.144 | ✗ | mixed | separated | 1.0 | 2.0 |
| 30 | 12.998 | ✓ | 0.323 | ✗ | mixed | separated | 1.0 | 2.0 |

Both Mode S windows have high KL scores (~13, far above the 5.42 threshold) but
low lang-id entropy (both < 0.38, since Mode S is monoscript Chinese). The OR
ensemble catches both via KL (saves 1.0 cpWER each vs always-separated). The
AND ensemble misses both (lang-id doesn't flag), routing them to separated
(cpWER 2.0 each) — identical to lang-id-alone.

### Paired-delta bootstrap CIs (per-window)

| Comparison | point Δ | CI 95% | excludes 0? |
|---|---:|:---:|:---:|
| OR − KL-alone | +0.000000 | [+0.000000, +0.000000] | tied (zero-width) |
| OR − lang-id-alone | −0.012987 | [−0.064935, +0.025974] | NO |
| OR − always-mixed | −0.142857 | [−0.311688, −0.012987] | **YES** |
| AND − KL-alone | +0.012987 | [−0.025974, +0.064935] | NO |
| AND − lang-id-alone | +0.000000 | [+0.000000, +0.000000] | tied (zero-width) |

The OR − KL paired-delta CI is **zero-width** [+0.000000, +0.000000]: OR and KL
produce identical per-window cpWER for all 77 windows. This is the strongest
possible evidence that the OR ensemble is cpWER-redundant with KL-alone. The
AND − lang-id CI is similarly zero-width.

OR significantly beats always-mixed per-window (upper CI −0.0130 < 0) — the
same per-window significance as KL-alone (RQ58), since OR == KL per-window.

### Decision counts

| Router | mixed | separated |
|---|---:|---:|
| KL-alone (RQ58) | 41 | 36 |
| lang-id-alone (RQ16) | 38 | 39 |
| OR ensemble | 43 | 34 |
| AND ensemble | 36 | 41 |

OR routes 2 more windows to mixed than KL-alone (windows 19, 44 — the
cpWER-tied FPs). AND routes 2 fewer windows to mixed than lang-id-alone (the
same 2 windows, removed by the AND intersection).

## Hypothesis Verdicts

### H60a — OR ensemble cpWER < 1.030 (beats KL-alone): KILLED

OR ensemble cpWER = 1.030303 >= kill threshold 1.030. The OR ensemble does NOT
beat KL-alone — it is **cpWER-identical** (delta = +0.000000, zero-width
paired-delta CI). The structural reason: KL-alone already catches 100% of all
37 hallucinations (including both Mode S windows), so the OR ensemble cannot
add any true positives. Its 2 additional lang-id flags (windows 19, 44) are
false positives on cpWER-tied windows (mixed == separated == 1.0), so they
change the decision but not the cpWER. The ensemble is cpWER-redundant with
KL-alone.

### H60b — OR catches 100% Mode S AND >= 90% all hallucinations: SUPPORTED

OR Mode S sensitivity = 100% (2/2); all-hallucination sensitivity = 100.0%
(37/37). The OR ensemble catches every hallucination KL catches (which is all
of them) plus 2 additional FPs. This is the maximum achievable sensitivity —
the OR ensemble is a strict superset of KL-alone's flags. H60b is supported
because the OR ensemble inherits KL's 100% Mode S sensitivity and 100%
all-hallucination sensitivity.

### H60c — AND ensemble FP rate < 7.5% (3/40): SUPPORTED

AND FP rate = 2.5% (1/40) < 7.5% (3/40). The AND ensemble has the lowest
false-positive rate of any detector tested: 1 FP (window 52, where both KL and
lang-id agree on a false positive), vs KL-alone's 4 FPs and lang-id-alone's 3
FPs. The AND intersection removes 2 of lang-id's 3 FPs (windows 19, 44 — where
lang-id flags but KL does not) and 3 of KL's 4 FPs (where KL flags but lang-id
does not), leaving only the 1 window where both detectors falsely agree.
H60c is supported. However, this FP reduction comes at the cost of missing
Mode S (AND routes windows 22 and 30 to separated, cpWER 2.0 each), so AND's
cpWER (1.043) matches lang-id-alone, not KL-alone.

## Why the ensemble doesn't help (the structural reason)

The key insight is that **KL-alone is already at the sensitivity ceiling** (100%
of all hallucinations, including Mode S). When one detector catches everything,
an OR ensemble can only add false positives — it cannot add true positives
because there are no missed hallucinations to recover. And when the additional
false positives happen to be cpWER-neutral (mixed == separated), the OR
ensemble's cpWER is identical to the better detector alone.

The AND ensemble has the opposite problem: it trades sensitivity for
specificity. By requiring both detectors to agree, it removes false positives
(1 FP vs 3–4) but also removes true positives — specifically Mode S, where KL
flags but lang-id does not. The AND ensemble's cpWER matches lang-id-alone (the
worse detector) because it inherits lang-id's Mode S blindness.

The two detectors ARE complementary in detection (KL catches Mode S, lang-id
catches diverse), but this complementarity does NOT translate to a cpWER
improvement because:
1. KL already catches the diverse hallucinations that lang-id catches (KL's
   100% sensitivity includes the 35 diverse hallucinations).
2. The only windows where the detectors disagree are 2 cpWER-tied false
   positives where the routing choice doesn't matter.

An ensemble would help if KL MISSED some hallucinations that lang-id catches
(and vice versa), with those missed hallucinations having mixed < separated
(so recovering them improves cpWER). But KL catches everything, so there's
nothing for lang-id to add. The complementarity is one-directional: lang-id
adds nothing to KL's coverage, while KL adds Mode S to lang-id's coverage —
but the OR ensemble already includes KL, so it gets Mode S for free.

## Honest Limitations

1. **In-sample calibration (inherited from RQ58/RQ16).** The KL threshold
   (5.418144) and lang-id threshold (0.38) are calibrated on these exact 77
   AISHELL-4 windows. The 90% specificity and 100% sensitivity are in-sample
   estimates. A proper test needs a held-out AISHELL-4 session.

2. **Single meeting, 77 windows (inherited).** Only `M_R003S02C01` is
   available. The bootstrap CI is over 77 windows, not over meetings. The
   lumpy, discrete word-level cpWER distribution (69 of 77 windows tie at 1.0)
   means the ensemble's effect is concentrated in a few windows.

3. **lang-id threshold discrepancy.** The task spec pre-registers lang-id
   threshold 0.38; RQ16/RQ13 use 0.409. We verified these give IDENTICAL
   results (no lang-id score falls between 0.38 and 0.409), so the choice is
   immaterial. Both reproduce RQ16's 1.043290 with 3/40 FPs and 94.6%
   sensitivity.

4. **KL threshold is fixed, not re-calibrated.** RQ60 uses RQ58's calibrated
   threshold 5.418144 as a FIXED input (per the task spec "threshold 5.42"),
   not re-calibrating. This is the correct pre-registered protocol (the
   threshold was calibrated on RQ58, not re-tuned on RQ60).

5. **cpWER is utterance-level (inherited from RQ16/RQ30).** Each speaker's
   whole Chinese string is one token. The cpWER-tied windows (mixed ==
   separated == 1.0) are an artefact of this granularity; at char-level, some
   ties might break. A char-level re-validation is the required follow-up.

6. **The ensemble's cpWER-neutrality is specific to this dataset.** The 2
   windows where OR and KL disagree (19, 44) happen to be cpWER-tied. On a
   different dataset, the additional OR false positives might have mixed >
   separated (worsening cpWER) or mixed < separated (improving cpWER). The
   ensemble's effect is data-dependent and not guaranteed to be neutral.

7. **No deployable routing input (inherited).** Per the project's hard safety
   rules, cpWER / references are not used as routing input — both detectors
   are computed only from the hypothesis transcripts (character n-grams and
   script categories), which is the deployable signal surface.

## What this changes for the project

1. **The ensemble is cpWER-redundant with KL-alone.** RQ58's KL-alone router
   (cpWER 1.030) remains the best corrected router on AISHELL-4. The OR
   ensemble doesn't improve on it because KL already catches 100% of
   hallucinations. The AND ensemble is worse (1.043, misses Mode S). RQ60
   confirms that KL-alone is the right single-detector choice — ensembling
   with lang-id adds complexity without cpWER benefit.

2. **The AND ensemble has the lowest FP rate (1/40 = 2.5%).** While its cpWER
   (1.043) is worse than KL-alone (1.030), its FP rate is the lowest of any
   detector. In a deployment scenario where false alarms are costly (e.g.,
   unnecessary mixed-route computation), the AND ensemble offers a
   specificity-first alternative — at the cost of missing Mode S. This is a
   useful design point on the sensitivity-specificity Pareto frontier.

3. **The complementarity is structural, not cpWER-improving.** KL and lang-id
   are complementary in WHAT they catch (Mode S vs diverse), but since KL
   catches everything, the complementarity doesn't translate to a cpWER
   improvement. Future ensembles should explore detectors that catch
   DIFFERENT hallucinations that KL misses (if any exist) — not detectors
   whose catches are a subset of KL's.

4. **The cpWER-tie structure limits ensemble gains.** 69 of 77 windows have
   mixed == separated == 1.0 (cpWER ties). Ensembles can only improve cpWER
   on the ~8 windows where mixed ≠ separated. Of these, KL already catches
   all the hallucinated ones. The remaining non-hallucinated windows where
   the detectors disagree (19, 44) are cpWER-tied, so ensemble routing
   changes are neutral. This is a fundamental limit of the word-level cpWER
   granularity on this dataset.

## Reproducibility

- Script: `results/frontier/ensemble_corrected_router/ensemble_router_analysis.py`
  (deterministic; numpy + scipy + stdlib; no Whisper / no LLM / no ollama).
- Tests: `tests/test_ensemble_router.py` (113 tests, pure helpers only — KL
  primitives from RQ34, lang-id from RQ13, BCa framework from RQ39, new RQ60
  ensemble helpers, end-to-end integration on synthetic data).
- Per-window data: `results/frontier/ensemble_corrected_router/ensemble_router_results.csv`
  (77 rows; KL score, lang-id entropy, kl_flag, lang_id_flag, or_flag,
  and_flag, kl_decision, lang_id_decision, or_decision, and_decision, and
  per-route cpWER for each).
- Summary + hypothesis verdicts: `results/frontier/ensemble_corrected_router/ensemble_router_results.json`
- Bootstrap: 10,000 resamples, seed=42, alpha=0.05. BCa uses jackknife
  acceleration; paired-delta uses the same resample indices for both arms.
- KL config: n=2 (bigram), reference = average 2-gram distribution of 40
  non-hallucinated tracks' concatenated separated text (vocab 1487),
  aggregation = MAX across per-speaker tracks.
- Thresholds: KL = 5.418144 (RQ58's calibrated, fixed for RQ60); lang-id =
  0.38 (task spec; identical results to RQ16's 0.409).
- Source data: `results/external_sanity_check/aishell4/rq1_aishell4_validation_results.json`
  (label `external/sanity-check`, read-only — not modified).
- Run: `/opt/homebrew/bin/python3 results/frontier/ensemble_corrected_router/ensemble_router_analysis.py`
  (~5 s; numpy + scipy only).
