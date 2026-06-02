# Project State

This document is for future Codex / AI coding agents so they can resume work without losing project context.

## Current Project Title

**When Should We Separate? Adaptive Routing and Speaker-Aware Evaluation for Overlapping Speech ASR**

## Completed Stages

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

### Average

- `fixed mixed: 0.302093`
- `fixed separated: 0.191846`
- `fixed cleaned: 0.181681`
- `router_v2: 0.120042`
- `oracle best: 0.120042`

## Error Type Findings

- LightOverlap separated degradation is mainly insertion + repetition.
- MidOverlap separated degradation is also insertion + repetition.
- HeavyOverlap / OppositeOverlap benefit strongly from separation.
- Duplicate suppression helps Light/Mid but is not universally better.

## Speaker-Aware Findings

- `speaker_macro_cer` shows whether the ASR preserves content for each speaker.
- `separated_whisper` and `separated_whisper_cleaned` are both better measured with speaker-aware metrics than with plain CER alone.

### speaker_macro_cer

- `NoOverlap separated: 0.054312, cleaned: 0.089278`
- `LightOverlap separated: 0.194170, cleaned: 0.135164`
- `MidOverlap separated: 0.175908, cleaned: 0.168620`
- `HeavyOverlap separated: 0.110821, cleaned: 0.146535`
- `OppositeOverlap separated: 0.047479, cleaned: 0.083193`

## cpCER-lite Findings

- No obvious speaker permutation mismatch was found in the verified gold cases.
- `speaker_assignment_gap = 0.0` for all five gold cases.
- Main errors are content-level, not speaker-swap-level.

## Synthetic Benchmark Findings

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
- The largest improvement comes from `SyntheticNoOverlap`.
- Synthetic results are silver robustness evidence, not gold evaluation.

## Risk-Aware Selector Findings

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

## How to Resume Work

Common commands:

```powershell
python -m src.adaptive_router_v2
python -m src.evaluate_error_types --case all
python -m src.evaluate_speaker_cer --case all
python -m src.evaluate_cpcer_lite --case all
python -m src.risk_aware_selector --case all
python -m src.router_ablation
python -m src.router_ablation_split
```

## Notes for Future Agents

- Do not use ground-truth CER or reference transcripts as routing input.
- References and CER are for evaluation only.
- Keep gold and synthetic evaluation clearly separated.
- Prefer adding new outputs over overwriting existing benchmark files.
