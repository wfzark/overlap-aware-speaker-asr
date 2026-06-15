# Multi-Tier Cascade: Pareto Frontier Analysis

| Strategy | Avg CER | Avg Cost | Frontier Status |
| --- | ---: | ---: | --- |
| fixed_mixed_small | 0.302093 | 1.00 | **PARETO** |
| fixed_separated_small | 0.191846 | 2.00 | dominated |
| fixed_separated_small_cleaned | 0.181681 | 2.10 | dominated |
| router_v2_tier1_only | 0.120042 | 1.60 | **PARETO** |
| router_v2_tier2_cascade | 0.11404 | 6.40 | **PARETO** |
| router_v2_tier3_cascade | 0.11404 | 19.20 | dominated |
| cost_first_cascade | 0.120042 | 1.60 | **PARETO** |
| accuracy_first_cascade | 0.11404 | 19.20 | dominated |
| oracle_best | 0.120042 | 999.00 | dominated |

## Deployment Recommendations

- **Cost-first**: `fixed_mixed_small` (CER=0.302093, Cost=1.00)
- **Accuracy-first**: `router_v2_tier2_cascade` (CER=0.11404, Cost=6.40)

**Note:** This Pareto analysis uses conservative CER estimates. Refresh with real larger-model measurements for production.