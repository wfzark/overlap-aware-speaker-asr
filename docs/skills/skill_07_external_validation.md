# Skill 07: External Mini Validation

## What question does this skill explore?

Can a tiny external sanity-check confirm that the current overlap-aware ASR story still makes sense outside the project's own gold and synthetic benchmarks?

## Why is it relevant to the current project?

The repository has a stable gold benchmark, silver synthetic validation, and multiple frontier coordination layers. A tiny external check helps test whether the current narrative survives one small out-of-domain slice without pretending to be a full benchmark.

## Current queue position

Second in the current frontier queue. Use it after the MeetEval bridge is clear enough to avoid confusion between compatibility work and external sanity-check work.

## Challenge level

Level 3: Research Extension

## Minimum viable attempt

- Pick one tiny slice from a documented external dataset
- Record source, license, and preprocessing steps
- Compare one or two current transcript types against the slice

## Stretch goal

- Compare the tiny external slice with the internal gold and synthetic views
- Document whether the same selective-separation story still holds
- Turn the slice into a repeatable sanity-check template

## Failure is useful if...

- The slice is too small to support strong claims, but it still exposes a useful domain shift
- The preprocessing burden is clear, even if the external result is weak
- The work shows that the project needs a narrower external setup before any larger benchmark attempt

## Inputs

- External dataset slice
- Internal transcript outputs
- Documentation on source and license

## Outputs

- External sanity-check note
- Small comparison table
- Explicit source and preprocessing record

## What not to do

- Do not present the slice as a completed external benchmark.
- Do not blur the line between sanity-check and evaluation.
- Do not lose the gold/silver/frontier labels.
- Do not use the external slice to rewrite the stable baseline conclusions without strong evidence.

## Success criteria

- A small external slice is documented clearly enough to repeat.
- The result is labeled as `external/sanity-check`.
- The note helps a future agent decide whether a larger external validation is worth the effort.

## Suggested agent prompt

Stage a tiny external sanity-check and document everything needed to repeat it. Keep the claim scope narrow and treat the result as a bridge, not a benchmark victory.

## Owner suggestion

Evaluation / dataset bridge owner.
