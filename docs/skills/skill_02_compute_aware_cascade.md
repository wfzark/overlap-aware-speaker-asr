# Skill 02: Compute-aware Cascaded Recognition

## What question does this skill explore?

When should the system spend more compute?

## Why is it relevant to the current project?

The current pipeline already has cheap and more expensive paths. This skill frames the problem as an accuracy-cost trade-off rather than a single best model chase.

## Challenge level

Level 3: Research Extension

## Minimum viable attempt

- Define a cheap route and a fallback route
- Measure CER and runtime for each route
- Plot the trade-off curve

## Stretch goal

- Add a risk-triggered third tier with manual review or critic repair
- Measure coverage and strong-model call rate
- Compare multiple cascades on the same benchmark

## Failure is useful if...

- The cost savings are small but the failure mode is clearly quantified
- The cascade adds complexity without improving enough to justify it
- The analysis reveals which risk features are worth paying attention to

## Inputs

- Router outputs
- Runtime measurements
- CER results
- Risk-aware selection outputs

## Outputs

- `results/tables/cascade_performance.csv`
- `results/figures/cer_runtime_tradeoff.png`

## What not to do

- Do not assume the largest model is always the answer.
- Do not use test CER to tune a cascade rule.
- Do not add a new heavy ASR model without a clear need.
- Do not hide compute cost just because accuracy improves.

## Success criteria

- The cascade shows a measurable accuracy/runtime trade-off.
- The proposal can be explained in one page.
- Manual review is used only when needed.

## Suggested agent prompt

Build a compute-aware cascade that respects runtime budget. Keep the experiment labeled, document compute cost, and report if the accuracy gain is worth the extra spend.

## Owner suggestion

Systems / deployment owner.
