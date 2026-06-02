# Project Context for Future Agents

## Mission
This repository studies the question:

**When should we separate? Adaptive routing and speaker-aware evaluation for overlapping speech ASR.**

The core research question is:

> Speech separation is useful, but not universally beneficial.

We compare three output families:

1. `mixed_whisper`
2. `separated_whisper`
3. `separated_whisper_cleaned`

The goal is to decide when each output is the safest final choice for overlapping-speech ASR.

## Hard Rules for Agents

- Do not use ground-truth CER or `reference_transcripts.json` as routing input.
- References and CER are for evaluation only.
- Do not overwrite verified references unless explicitly requested.
- Do not rerun ASR unless explicitly requested.
- Do not modify existing result tables unless the requested stage requires recomputation.
- Do not introduce new heavy ASR models without approval.
- Do not treat synthetic silver results as gold evaluation.
- Keep gold and synthetic evaluations clearly separated.
- Prefer adding new result files over overwriting existing ones.

## Current Best Quantitative Result

### Gold benchmark

- `router_v2 average CER = 0.120042`
- `oracle best average CER = 0.120042`

### Synthetic original 25

- `v1 = 0.350902`
- `v2 = 0.167553`
- `oracle = 0.082239`

### Synthetic split held-out test

- `v1 = 0.361350`
- `v2 = 0.335326`
- `oracle = 0.115181`

## Current Interpretation

- `v1` looks perfect on gold but is unstable on synthetic.
- `v2` fixes part of the `v1` instability, especially `SyntheticNoOverlap`.
- Ablation shows repetition / removed_count are more useful than `length_ratio` alone.
- `cpCER-lite` shows no speaker permutation mismatch in gold cases.
- `risk_aware_selector` is a deployability and explainability layer, not the primary best-CER result.

## Important Result Hierarchy

### Gold benchmark

- 5 manually verified cases.
- Main quantitative result.

### Synthetic silver benchmark

- Supplementary robustness validation.
- Not gold evaluation.

### Held-out synthetic split

- Used to test router robustness.
- Do not tune rules on held-out test results.

## Completed Work

- Stage 1 project skeleton
- mixed Whisper baseline
- separated speaker-track ASR
- duplicate suppression
- 5 verified gold references
- global CER
- error type analysis
- adaptive router v1
- speaker-aware CER
- synthetic overlap benchmark
- synthetic silver evaluation
- router v2
- synthetic audit
- router ablation
- held-out synthetic split validation
- cpCER-lite
- risk-aware selector

## Gold Benchmark Final CER Table

### NoOverlap

- `mixed_whisper: 0.215827`
- `separated_whisper: 0.053957`
- `separated_whisper_cleaned: 0.089928`
- `best: separated_whisper`

### LightOverlap

- `mixed_whisper: 0.210714`
- `separated_whisper: 0.475000`
- `separated_whisper_cleaned: 0.382143`
- `best: mixed_whisper`

### MidOverlap

- `mixed_whisper: 0.178947`
- `separated_whisper: 0.273684`
- `separated_whisper_cleaned: 0.207018`
- `best: mixed_whisper`

### HeavyOverlap

- `mixed_whisper: 0.386861`
- `separated_whisper: 0.109489`
- `separated_whisper_cleaned: 0.145985`
- `best: separated_whisper`

### OppositeOverlap

- `mixed_whisper: 0.518116`
- `separated_whisper: 0.047101`
- `separated_whisper_cleaned: 0.083333`
- `best: separated_whisper`

### Averages

- `fixed mixed: 0.302093`
- `fixed separated: 0.191846`
- `fixed cleaned: 0.181681`
- `router_v2: 0.120042`
- `oracle best: 0.120042`

## Error Type Findings

- `LightOverlap` separated degradation is mainly insertion + repetition.
- `MidOverlap` separated degradation is also insertion + repetition.
- `HeavyOverlap` and `OppositeOverlap` benefit strongly from separation.
- Duplicate suppression helps Light/Mid but is not universally better.

## Speaker-Aware Findings

### speaker_macro_cer

- `NoOverlap separated: 0.054312, cleaned: 0.089278`
- `LightOverlap separated: 0.194170, cleaned: 0.135164`
- `MidOverlap separated: 0.175908, cleaned: 0.168620`
- `HeavyOverlap separated: 0.110821, cleaned: 0.146535`
- `OppositeOverlap separated: 0.047479, cleaned: 0.083193`

## cpCER-lite

- No obvious speaker permutation mismatch.
- `speaker_assignment_gap = 0.0` for all five gold cases.
- Main errors are content-level, not speaker-swap-level.

## Synthetic Findings

### Original 25 synthetic silver

- `v1: 0.350902`
- `v2: 0.167553`
- `oracle: 0.082239`

### Held-out synthetic test

- `v1: 0.361350`
- `v2: 0.335326`
- `oracle: 0.115181`

### Interpretation

- `v2` improves stability but still has a large gap to oracle.
- Improvement mainly comes from `SyntheticNoOverlap`.
- Synthetic results are silver robustness evidence, not gold evaluation.

## Risk-Aware Selector Findings

### Gold final selection

- `NoOverlap -> separated_whisper`
- `LightOverlap -> mixed_whisper`
- `MidOverlap -> mixed_whisper`
- `HeavyOverlap -> separated_whisper_cleaned`
- `OppositeOverlap -> separated_whisper_cleaned`

### Averages

- `risk_aware_selector: 0.134587`
- `router_v2: 0.120042`
- `oracle_best: 0.120042`

### Interpretation

- Risk-aware selector is more conservative and explainable.
- It is not the primary best-CER result.
- It provides reference-free risk detection and selective repair.

## Recommended Next Stages

1. Update `REPORT.md` and `README.md` if needed.
2. Create a Streamlit demo.
3. Create a final presentation script.
4. Optional: external mini validation or MeetEval compatibility discussion.

## Commands Commonly Used to Resume Work

```powershell
python -m src.adaptive_router_v2
python -m src.evaluate_error_types --case all
python -m src.evaluate_speaker_cer --case all
python -m src.evaluate_cpcer_lite --case all
python -m src.risk_aware_selector --case all
python -m src.router_ablation
python -m src.router_ablation_split
```
