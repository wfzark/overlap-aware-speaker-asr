# RQ62 — Cascade with KL+lang-id Ensemble Gate

**Label:** experimental/frontier
**Issue:** #986
**Builds on:** RQ13 (lang-id entropy, PR #904), RQ43 (3-tier KL cascade, PR #959), RQ44 (OOB bootstrap + lang-id cal, PR #963), RQ46 (original-rule CI anchor, PR #966), RQ48 (`count_modes`, PR #965), RQ54 (F1 cascade comparison, PR #971), RQ58 (2-gram KL detector + cal, PR #975), RQ59 (BCa CI + jackknife framework, PR #980)

## Research question

RQ59 showed that on the KL detector, both **Youden's J** and **F1** calibration collapse to the same aggressive operating point (KL threshold = 0.01, 83.1% escalation, OOB cpWER ~0.78) because the KL ROC is **flat-topped**: raising the threshold above 0.01 loses hallucinated windows (sensitivity falls) without gaining specificity (clean high-KL windows stay flagged). RQ62 asks whether a **KL+lang-id ensemble gate** produces a less aggressive — or more robust — cascade operating point than the single-KL gate.

## Pre-registered hypotheses

| ID | Statement | Kill | Primary gate |
|---|---|---|---|
| H62a | Ensemble cascade escalates < 83.1% of windows to base | escalation ≥ 0.831 | OR |
| H62b | Ensemble cascade OOB cpWER ≤ 0.889 (matches RQ43's 0.888947) | median > 0.889 | OR |
| H62c | Ensemble cascade BCa CI width ≤ 0.2489 (RQ46's original-rule width) | width > 0.2489 | OR |

## Method (REANALYSIS ONLY — no Whisper / no LLM)

The cascade simulation is **held fixed** at RQ43's actual implementation (real whisper-tiny separated cpWER for tier 1; `base = tiny * 0.428031` for tier 3) so the comparison to RQ43/RQ46/RQ54/RQ59 is apples-to-apples. The **only** independent variable vs RQ59 is the gate: RQ59's single-KL Youden's J is replaced by a two-detector ensemble.

1. Load RQ43's 77 per-window `(tiny_sep_cpwer, base_sep_cpwer, kl_sep, window_id)` and RQ58's 77 per-window `(kl_score, lang_id_entropy, window_id)`; join by `window_id` (both are `0..76` in order, verified in tests).
2. Hallucination label: `tiny_sep_cpwer > 1.0` (37 hallucinated / 40 clean — same label rule as RQ44/RQ48/RQ54/RQ59).
3. In-sample calibration (BOTH detectors at ≥ 90% specificity on the 40 clean windows):
   - **KL** threshold via RQ58's candidate-set rule → **5.418144** (reproduces RQ58 bit-for-bit).
   - **lang-id** threshold via RQ44's 0.01-step grid → **0.38** (reproduces RQ44 bit-for-bit).
4. In-sample ensemble gate (OR and AND) at the calibrated thresholds → cascade cpWER, escalation fraction, compute.
5. Bootstrap B=10000, seed=42: re-calibrate BOTH thresholds on in-bag windows, evaluate ensemble cascade cpWER (OR and AND) on OOB windows (RQ44's OOB protocol).
6. Delete-1 jackknife (77 fits) for the BCa acceleration, separately for OR and AND.
7. BCa 95% CI on the OOB cpWER distribution (Acklam inverse-normal, no scipy), separately for OR and AND.
8. Mode count on the KL-threshold and lang-id-threshold bootstrap distributions (RQ48's `count_modes`, min_fraction=0.05).

## Headline results

### In-sample ensemble (n=77)

| Gate | KL thr | lang thr | Escalated | Fraction | Cascade cpWER | Compute |
|---|---|---|---|---|---|---|
| KL alone (RQ58) | 5.418144 | — | 41/77 | 0.5325 | — | — |
| lang alone (RQ44) | — | 0.38 | 38/77 | 0.4935 | — | — |
| **OR** (either flags) | 5.418144 | 0.38 | **43/77** | **0.5584** | **0.9335** | **1.519x** |
| **AND** (both flag) | 5.418144 | 0.38 | **36/77** | **0.4675** | **1.0004** | **1.435x** |
| RQ59 Youden's J (KL alone) | 0.01 | — | 64/77 | 0.8312 | 0.7824* | 1.861x |
| RQ54 F1 (KL alone) | 0.01 | — | 64/77 | 0.8312 | 0.7799* | 1.861x |

*RQ59/RQ54 cpWER is the OOB median, not in-sample — these gates' in-sample cpWER is also lower than the ensemble's because they escalate 83.1% of windows to base (cheap base ratio hides hallucinations). The ensemble escalates far fewer windows, so its in-sample cpWER is higher but its OOB cpWER (after re-calibration on in-bag windows) is the fair comparison.

### Bootstrap OOB cpWER (B=10000, seed=42)

| Gate | OOB median | OOB mean | 2.5% pct | 97.5% pct | BCa 95% CI | BCa width | Modes (KL / lang) |
|---|---|---|---|---|---|---|---|
| **OR** | **0.9423** | 0.9485 | 0.8430 | 1.1142 | [0.8300, 1.0691] | **0.2391** | 7 / 5 |
| **AND** | **1.0172** | 1.0662 | 0.9066 | 1.6293 | [0.8909, 1.5061] | **0.6153** | 7 / 5 |

### Hypothesis verdicts (primary gate = OR)

| ID | Statement | OR verdict | AND verdict | Reference |
|---|---|---|---|---|
| **H62a** | Escalation < 83.1% | **SUPPORTED** (0.5584 < 0.831) | SUPPORTED (0.4675 < 0.831) | RQ54/RQ59 = 0.8312 |
| **H62b** | OOB cpWER ≤ 0.889 | **KILLED** (median 0.9423 > 0.889) | KILLED (median 1.0172 > 0.889) | RQ43 = 0.888947 |
| **H62c** | BCa width ≤ 0.2489 | **SUPPORTED** (0.2391 ≤ 0.2489) | KILLED (0.6153 > 0.2489) | RQ46 = 0.2489 |

**Overall:** 2 of 3 hypotheses supported on the primary (OR) gate. The ensemble cascade **escapes the 83.1% escalation collapse** that plagued RQ59's Youden's J and RQ54's F1, and the OR gate **maintains the RQ46 CI-width robustness** — but it does so at the cost of a higher OOB cpWER (0.9423 vs RQ43's 0.889), killing H62b.

## Mechanism: why the ensemble escapes the flat-topped-ROC collapse

RQ59's collapse was a **calibration-rule artifact, not a KL-detector artifact**. Youden's J and F1 on RQ43's `kl_sep` (n=3 KL, range `[0, 8.5255]`) both picked threshold 0.01 — the lowest grid point — because the KL ROC is flat-topped: the cleanest operating point (sens=1.0, spec=0.325) is achieved across `[0.01, ~2.98]` and the lowest-threshold tie-break picks 0.01, escalating 83.1% of windows.

RQ62 sidesteps this in two ways:

1. **Different KL calibration rule.** RQ58's `calibrate_threshold_at_specificity` is a **candidate-set, ≥ 90% specificity** rule (not a Youden's-J / F1 max rule). It picks the smallest KL threshold that holds the clean-window false-positive count to ≤ 4 of 40. On the 2-gram KL detector (range `[0, 20.72]`) this gives threshold **5.418144**, escalating only 41/77 = 53.25% of windows — far below 83.1%. The 2-gram KL detector is also a *different* detector from RQ43's n=3 KL: its threshold is higher because its score distribution has more headroom.
2. **Second detector + ensemble logic.** Adding lang-id (threshold 0.38, escalating 38/77 = 49.35% alone) and combining via OR yields 43/77 = 55.84% — only 2 windows above the lang-id threshold but below the KL threshold. The two detectors overlap heavily (35 of 37 hallucinations are flagged by both). The AND gate, conversely, escalates only the 36 windows both detectors flag.

The ensemble does not "fix" the flat-topped ROC — it bypasses it by using a calibration rule (≥ 90% specificity) that does not optimize a J/F1 criterion, on a different KL detector (2-gram instead of n=3), with a second orthogonal signal. The result is a gate that escalates far fewer windows.

## Why H62b fails: less aggressive ≠ more accurate

The OR gate's in-sample cpWER is **0.9335**, vs RQ59's in-sample 0.7824. This is not a bug — it's the trade-off the ensemble makes:

- RQ59's 83.1% escalation sends 64 of 77 windows to base (`cpWER = tiny * 0.428031`). For hallucinated windows (tiny > 1.0), this is a large win — base is 57% lower. For clean windows (tiny ≤ 1.0), this is a small loss — base has the same relative ratio but no hallucination to correct.
- RQ62's OR gate escalates only 43/77. The 21 non-escalated windows stay at tiny. Of those, the hallucinated ones (where tiny > 1.0) contribute their full hallucinated cpWER to the cascade mean.

Because RQ58's KL threshold (5.42) is calibrated to ≥ 90% specificity, it deliberately *misses* some hallucinations to keep the false-positive count low. The ensemble inherits this trade-off. The OOB median cpWER (0.9423) is therefore *above* RQ43's 0.889 — the ensemble is less aggressive but also less accurate on the catastrophic hallucinations it leaves at tier 1.

## Why H62c holds on OR but fails on AND

The OR gate's BCa CI width (0.2391) is slightly *narrower* than RQ46's 0.2489 anchor — surprising, given that the ensemble has MORE moving parts (two re-calibrated thresholds per resample). The reason: the OR gate's OOB cpWER distribution is more concentrated (IQR ≈ 0.27) than the AND gate's (IQR ≈ 0.72). The OR gate escalates 43/77 windows, so the OOB subset (mean size 28.2) typically contains both escalated and non-escalated windows — the cpWER mean is stable across resamples.

The AND gate's CI width (0.6153) is 2.6× the OR width. The AND gate escalates only 36/77, and the lang-id threshold has 5 modes — including a 9% mode at 0.01 (the lowest grid point) that escalates everything. When a bootstrap resample picks this mode, the AND-gate cpWER swings to the all-base mean (0.428 × tiny_mean); when it picks 0.87 (the second-most-common mode), it escalates almost nothing. This bimodality inflates the OOB cpWER variance and widens the BCa CI.

## KL threshold bootstrap distribution: 7 modes, 8.3% inf

The KL threshold distribution has 7 modes at ≥ 5% frequency:
- 6.5539 (23.6%), 5.4181 (17.5%), 12.3464 (16.5%), 5.4037 (11.7%), **inf (8.3%)**, 4.5357 (6.6%), 12.9984 (6.3%).

The `inf` mode (830 of 10000 resamples) occurs when the in-bag subset has too few negative samples to satisfy the ≥ 90% specificity floor at any finite threshold — RQ58's calibration returns `inf` in this case. Tests verify `inf` thresholds propagate correctly through the gate logic (inf KL threshold → no KL flags → OR gate falls back to lang-id alone; AND gate escalates nothing). The BCa CI is on the OOB cpWER (not the threshold), so the inf thresholds do not break the CI computation.

The lang-id threshold distribution is more stable: 5 modes, with 0.38 (the in-sample threshold) dominating at 60.4%.

## Compute savings

| Gate | Compute (× tiny) | vs RQ59 |
|---|---|---|
| RQ59 Youden's J | 1.861x | — |
| **RQ62 OR** | **1.519x** | **−18.4%** |
| **RQ62 AND** | **1.435x** | **−22.9%** |

The ensemble saves 18–23% compute over RQ59 by escalating fewer windows to the 1.93× base model — at the cost of a higher cpWER.

## Controlled-comparison caveats

1. **RQ46's 0.2489 anchor is a percentile CI evaluated in-bag at the FIXED threshold 3.30.** RQ62's BCa CI is bias-corrected + accelerated and evaluated OOB at RE-CALIBRATED ensemble thresholds. The H62c comparison is therefore directional (does the ensemble + BCa + OOB keep the interval within the original-rule width), not a pure like-for-like CI-method swap. This mirrors RQ54's H54b and RQ59's H59c caveat.
2. **RQ59's 0.7824 OOB cpWER is not directly comparable to RQ62's 0.9423.** RQ59 escalates 83.1% of windows to base; RQ62-OR escalates 55.8%. RQ59's lower cpWER is bought with 1.86× compute; RQ62's higher cpWER is bought with 1.52× compute. The fair comparison is on the **Pareto frontier** of (cpWER, compute), not cpWER alone.
3. **The KL detector differs between RQ59 and RQ62.** RQ59 uses RQ43's `kl_sep` (n=3 KL, range `[0, 8.5255]`); RQ62 uses RQ58's 2-gram character KL (range `[0, 20.72]`). This is by design — the issue specifies RQ58's 2-gram KL with threshold 5.42 — but it means the ensemble is not a pure "add lang-id to RQ59" experiment. The 2-gram KL's higher threshold (5.42 vs 0.01) is the primary driver of the lower escalation.

## Files

- `ensemble_cascade_analysis.py` — main analysis script (numpy + stdlib only; scipy / sklearn / Whisper NOT required).
- `ensemble_cascade_results.json` — full results (in-sample, bootstrap, BCa CI, jackknife, hypothesis verdicts, per-bootstrap arrays).
- `ensemble_cascade_results.csv` — per-bootstrap (kl_thr, lang_thr, or_oob_cpwer, and_oob_cpwer, n_oob).
- `FINDINGS.md` — this file.
- `tests/test_ensemble_cascade.py` — 108 tests pinning constants, data loading, calibration, gate logic, cascade simulation, in-sample ensemble, bootstrap, jackknife, BCa helpers, count_modes, finite-stats helper, output files, and hypothesis verdicts.

## Reproduce

```bash
cd <repo-root>
/opt/homebrew/bin/python3 results/frontier/ensemble_cascade_gate/ensemble_cascade_analysis.py
/opt/homebrew/bin/python3 -m unittest tests.test_ensemble_cascade -v
```

No network, no Whisper, no LLM. Runs in ~3 seconds on a 2023 MacBook Pro.
