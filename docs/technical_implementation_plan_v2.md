# When Should We Separate? Adaptive Routing and Speaker-Aware Evaluation for Overlapping Speech ASR

## Project Pivot Summary

This project started from the earlier idea of "Overlap-Aware Speaker-Attributed ASR with RAG-Enhanced LLM Correction". After completing the benchmark pipeline and comparing mixed ASR, separated ASR, and duplicate-suppressed separated transcripts, the core research direction has been adjusted.

The current core question is:

> When should we separate overlapping speech, and when does separation hurt more than it helps?

LLM/RAG remains a possible future extension or optional demo path, but it is no longer the core experimental line.

## Current Completed Work

- GitHub repository created and maintained with commits.
- Five benchmark cases prepared:
  - NoOverlap
  - LightOverlap
  - MidOverlap
  - HeavyOverlap
  - OppositeOverlap
- Five verified reference transcripts created.
- Whisper small baseline completed for mixed audio.
- Whisper small baseline completed for separated speaker tracks.
- Duplicate suppression post-processing completed for separated transcripts.
- CER evaluation completed.
- Adaptive/oracle analysis figures and tables generated.

## Key Experimental Findings

### Case-Level CER Summary

| case_id | mixed_cer | separated_cer | separated_cleaned_cer | best_method |
| --- | ---: | ---: | ---: | --- |
| NoOverlap | 0.215827 | 0.053957 | 0.089928 | separated_whisper |
| LightOverlap | 0.210714 | 0.475000 | 0.382143 | mixed_whisper |
| MidOverlap | 0.178947 | 0.273684 | 0.207018 | mixed_whisper |
| HeavyOverlap | 0.386861 | 0.109489 | 0.145985 | separated_whisper |
| OppositeOverlap | 0.518116 | 0.047101 | 0.083333 | separated_whisper |

### Average CER

| method | average_cer |
| --- | ---: |
| mixed_whisper | 0.302093 |
| separated_whisper | 0.191846 |
| separated_whisper_cleaned | 0.181681 |
| oracle/adaptive best | 0.120042 |

### Interpretation

- Separated speaker-track ASR is clearly beneficial for NoOverlap, HeavyOverlap, and OppositeOverlap.
- Separated speaker-track ASR is worse than mixed ASR for LightOverlap and MidOverlap.
- Duplicate suppression helps reduce repeated hallucinations, but does not fully fix the failure mode.
- The strongest signal from the benchmark is not "always separate", but "separate selectively".

## Updated Core Research Questions

RQ1: Does speech separation always improve multi-speaker ASR?

RQ2: What error patterns are introduced by separated speaker-track ASR?

RQ3: Can a rule-based adaptive router select the better transcription path without using ground-truth CER?

RQ4: Does speaker-aware evaluation reveal behaviors that global CER misses?

## Updated Architecture

```text
audio case
  -> mixed_whisper
  -> separated_whisper
  -> separated_whisper_cleaned
  -> error type analysis
  -> adaptive router
  -> speaker-aware evaluation
```

## Next Implementation Stages

### Stage 10: Adaptive Router

Implement a routing policy that selects one of:

- mixed_whisper
- separated_whisper
- separated_whisper_cleaned

The router must not use ground-truth CER as an input feature.

### Stage 11: Error Type Analysis

Analyze repeated hallucinations, insertion errors, and tail duplication in separated transcripts.

### Stage 12: Speaker-Aware Evaluation

Extend evaluation beyond global CER to speaker-aware metrics and case-level diagnostics.

### Stage 13: Final Documentation

Finalize README, REPORT, and presentation materials.

## What We Will Not Do in Core Scope

- Train ASR models
- Train diarization models
- Modify Whisper architecture
- Continue expanding LLM/RAG as the central experimental line
- Build a complex frontend
- Add large-scale external benchmarks

## Final Expected Contribution

The final contribution is a compact but complete experimental framework showing that speech separation should be applied selectively. The adaptive routing idea is motivated by observed failure modes, and the evaluation emphasizes both global CER and speaker-aware diagnostics.
