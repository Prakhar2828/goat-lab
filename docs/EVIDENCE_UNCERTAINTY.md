# Evidence Uncertainty Policy

## Status

This document freezes the GOAT Lab v1 category-level uncertainty policy.
It does not freeze the production category scale or production weights, and
it does not permit the final simulation.

## Central-score rule

Uncertainty is represented by an interval around each existing category
score. The interval layer must not modify the central score. A category
score is therefore identical before and after uncertainty processing.

## Base intervals

Each category has a preregistered base half-width, coverage estimate, and
confidence estimate in `configs/evidence_uncertainty.json`. Lower coverage
or lower confidence widens the interval. Every interval is clipped to the
valid 0–100 category scale.

These intervals are methodological uncertainty bands. They are not claimed
to be frequentist confidence intervals or Bayesian credible intervals.

## Missing evidence

Missing evidence never becomes a zero. It is represented through lower
coverage, lower confidence, or a wider interval. The central score is not
penalized merely because an optional evidence family is unavailable.

## Defense evidence

When the structured defense-evidence audit is available, its coverage and
confidence fields replace the generic defense defaults. Defensive awards
may contribute to the central defense score through the separately frozen
defense policy. Film evidence remains governed by the expert policy below.

## Expert Film Consensus

Expert Film Consensus has a central-score weight of zero in v1. Existing
phase-level claims remain diagnostic. An offense or defense interval may
narrow only when corresponding rows are already marked
`PRIMARY_MODEL_ELIGIBLE` and meet the frozen independent-family threshold.

With the current evidence registry, there are zero primary-eligible rows.
Expert film therefore changes no central score and narrows no interval.

## Release gate

The audit must verify:

- two target players and nine categories;
- one interval per player-category pair;
- finite, ordered, bounded intervals;
- unchanged central scores;
- zero expert central-score weight;
- no expert narrowing without primary eligibility;
- zero release blockers;
- final simulation remains blocked.
