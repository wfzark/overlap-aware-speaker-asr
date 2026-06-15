# Multi-Tier Cascade: Per-Case Routing Decisions

| Case | Overlap | Risk | T1 Method | T1 CER | ->T2 | T2 CER | ->T3 | Final Tier | Final CER | Cost |
| --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| HeavyOverlap | 3 | medium | separated_whisper_small | 0.1095 | YES | 0.1040 | YES | 3 | 0.1040 | 24.0 |
| LightOverlap | 1 | high | mixed_whisper_small | 0.2107 | YES | 0.2002 | YES | 3 | 0.2002 | 12.0 |
| MidOverlap | 2 | high | mixed_whisper_small | 0.1789 | YES | 0.1700 | YES | 3 | 0.1700 | 12.0 |
| NoOverlap | 0 | medium | separated_whisper_small | 0.0540 | YES | 0.0513 | YES | 3 | 0.0513 | 24.0 |
| OppositeOverlap | 4 | medium | separated_whisper_small | 0.0471 | YES | 0.0447 | YES | 3 | 0.0447 | 24.0 |

## Risk Reasons

- **HeavyOverlap**: length_ratio=2.47>1.35
- **LightOverlap**: length_ratio=2.56>1.35; dup_removed=6>=5; overlap_level=1 (separation-hurts zone)
- **MidOverlap**: length_ratio=2.55>1.35; overlap_level=2 (separation-hurts zone)
- **NoOverlap**: length_ratio=2.17>1.35; no_overlap_many_segments=26
- **OppositeOverlap**: length_ratio=3.26>1.35