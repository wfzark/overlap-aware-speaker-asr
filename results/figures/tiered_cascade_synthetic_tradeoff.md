# Multi-Tier Cascade: CER-Runtime Tradeoff

| Strategy | Avg CER | Avg Cost | Rel Cost vs T1 | T2 Esc | T3 Esc | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_mixed_small | 0.311442 | 1.00 | 0.714286 | 0 | 0 | 100% |
| fixed_separated_small | 0.380701 | 2.00 | 1.428571 | 0 | 0 | 100% |
| fixed_separated_small_cleaned | 0.203778 | 2.10 | 1.5 | 0 | 0 | 100% |
| router_v2_tier1_only | 0.167553 | 1.40 | 1.0 | 0 | 0 | 100% |
| router_v2_tier2_cascade | 0.142694 | 5.60 | 4.0 | 25 | 0 | 100% |
| router_v2_tier3_cascade | 0.126212 | 16.80 | 12.0 | 25 | 25 | 100% |
| cost_first_cascade | 0.350902 | 1.60 | 1.142857 | 0 | 0 | 100% |
| accuracy_first_cascade | 0.126212 | 16.80 | 12.0 | 25 | 25 | 100% |
| oracle_best | 0.082239 | 999.00 | 713.571429 | 0 | 0 | 100% |

**Interpretation:**
- `router_v2_tier1_only`: Best cost-efficiency; all cases use cheap whisper-small.
- `router_v2_tier2_cascade`: Escalates high-risk cases to medium; better CER at moderate cost.
- `router_v2_tier3_cascade`: Escalates still-risky to large+LLM; best CER at highest cost.
- `cost_first_cascade`: Always pick cheapest viable route per case.
- `accuracy_first_cascade`: Always use strongest tier for every case.

**Note:** Tier 2/3 CER values are conservative ESTIMATES bounded by oracle CER.