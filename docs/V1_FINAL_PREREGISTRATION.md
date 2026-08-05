# Version 1 Final Preregistration

## Status

This document freezes the production model before the final 250,000-draw
simulation. No weights, scale choices, seeds, evidence eligibility rules, or
category definitions may be changed after the final result is observed.

## Production scale

The production scale is `bounded_logit_tail`.

It is applied to the six categories that are current historical-reference
percentiles:

- peak;
- prime;
- longevity;
- regular season;
- playoffs;
- offense.

The transform reduces extreme empirical-percentile saturation while
preserving category ordering and a bounded 0-100 score.

Defense is not transformed by the historical box-only scaling row. The
current defense score is a composite of historical box percentile evidence
and defensive-award evidence. It therefore remains on its current native
0-100 evidence scale. Winning context and cultural impact also remain on
their native evidence scales.

## Frozen hierarchy

Group masses are fixed:

- Performance Arc: 0.50
- Basketball Value: 0.40
- Broader Legacy: 0.10

The frozen central category weights are:

| Category | Weight |
|---|---:|
| Peak | 0.125 |
| Prime | 0.100 |
| Longevity | 0.075 |
| Regular season | 0.100 |
| Playoffs | 0.100 |
| Offense | 0.180 |
| Defense | 0.120 |
| Winning context | 0.100 |
| Cultural impact | 0.100 |

These weights are not selected in response to the final result.

## Final simulation

The final simulation is frozen at:

- 250,000 draws;
- random seed 23;
- within-group Dirichlet concentration 100.

Each draw keeps the three group masses exactly fixed. Only the relative
weights inside a multi-category group vary. The single-category Broader
Legacy group remains fixed.

This replaces the earlier flat nine-category Dirichlet simulation for the
version 1 release because a flat simulation can violate the preregistered
group caps.

## Evidence eligibility

The following remain diagnostic and carry zero additional central weight:

- expert film consensus;
- supporting-cast estimates;
- injury and roster-health gaps;
- locally available impact metrics;
- game-level playoff audit;
- cultural-impact sensitivity scenarios.

Expert-film release blockers remain visible as evidence-completeness
advisories. They do not block version 1 because no expert-film claim is
primary-model eligible and no expert-film value enters the central score.

## Release gate

The final simulation may run only when:

- hierarchy, group caps, production weights, and production scale are frozen;
- every mandatory audit reports zero release blockers;
- diagnostic patches report no central-score change and zero added weight
  where those fields apply;
- production category scores are complete, finite, and within 0-100;
- the target-player set is exactly Michael Jordan and LeBron James.

The gate produces a machine-readable check table and metadata report before
the final simulation is run.
