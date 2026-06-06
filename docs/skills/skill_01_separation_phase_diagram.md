# Skill 01: Separation Phase Diagram

## What question does this skill explore?

When does speech separation help, and when does it hurt?

## Why is it relevant to the current project?

The benchmark already shows that separation is helpful in some overlap regimes and harmful in others. This skill turns that observation into a systematic curve over overlap ratio.

## Current queue position

Not in the current frontier queue. Treat this as a later research-extension card after the current queue head and the other breadth-first handoffs are clearer.

## Challenge level

Level 3: Research Extension

## Minimum viable attempt

- Sweep a small set of overlap ratios
- Compute delta CER between separated and mixed outputs
- Plot the trend as a simple line chart

## Stretch goal

- Build a denser phase diagram over more tiers and more samples
- Identify distinct boundary regions where separation flips from helpful to harmful

## Failure is useful if...

- The curve is noisy but still reveals a boundary effect
- The experiment shows the current benchmark is too small and needs a larger controlled sweep
- The result explains why a single router rule cannot be universal

## Inputs

- Mixed audio cases with controlled overlap ratio
- Separated speaker-track outputs
- CER results for mixed and separated outputs

## Outputs

- `results/tables/separation_phase_diagram.csv`
- `results/figures/separation_phase_diagram.png`

## What not to do

- Do not replace the gold benchmark.
- Do not treat synthetic silver as gold.
- Do not adjust references just to make the curve look better.
- Do not call the output a universal separation law if it is only a local sweep.

## Success criteria

- The diagram shows a clear delta-CER trend across overlap ratios.
- `delta CER = CER(separated) - CER(mixed)` is interpretable.
- The plot helps explain why routing should be selective.

## Suggested agent prompt

Explore the boundary where separation helps and hurts. Keep gold/silver labels explicit, use controlled overlap sweeps, and report both success and failure modes.

## Owner suggestion

Research / evaluation owner.
