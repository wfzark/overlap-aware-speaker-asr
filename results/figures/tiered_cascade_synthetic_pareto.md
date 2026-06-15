# Multi-Tier Cascade: Pareto Frontier Analysis

| Strategy | Avg CER | Avg Cost | Frontier Status |
| --- | ---: | ---: | --- |
| fixed_mixed_small | 0.311442 | 1.00 | **PARETO** |
| fixed_separated_small | 0.380701 | 2.00 | dominated |
| fixed_separated_small_cleaned | 0.203778 | 2.10 | dominated |
| router_v2_tier1_only | 0.167553 | 1.40 | **PARETO** |
| router_v2_tier2_cascade | 0.142694 | 5.60 | **PARETO** |
| router_v2_tier3_cascade | 0.126212 | 16.80 | **PARETO** |
| cost_first_cascade | 0.350902 | 1.60 | dominated |
| accuracy_first_cascade | 0.126212 | 16.80 | **PARETO** |
| oracle_best | 0.082239 | 999.00 | **PARETO** |

## Deployment Recommendations

- **Cost-first**: `fixed_mixed_small` (CER=0.311442, Cost=1.00)
- **Accuracy-first**: `oracle_best` (CER=0.082239, Cost=999.00)

**Note:** This Pareto analysis uses conservative CER estimates. Refresh with real larger-model measurements for production.