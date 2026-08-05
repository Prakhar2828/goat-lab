# Impact Metric Evidence Policy

## Purpose

This audit determines which impact metrics are reproducibly available for Michael Jordan and LeBron James without treating incomplete or mismatched data as equivalent evidence.

## Central-score policy

Impact metrics audited in this patch receive zero additional central-score weight.

This prevents double counting because scoring, playmaking, defense, team results, and season value already incorporate overlapping information.

The existing category scores are not changed by this audit.

## Available local metrics

The local datasets contain partial coverage for:

- BPM
- VORP
- WS per 48
- net rating
- PIE
- raw plus-minus

These metrics are retained as diagnostic evidence only because their coverage differs substantially by player and era.

## Unsupported metrics

The following metrics are not available in a reproducible local dataset:

- genuine player on-off net rating
- RAPM
- EPM
- LEBRON

They are recorded as unavailable and are not estimated or substituted.

## Net rating is not on-off

Player net rating describes team performance during minutes associated with the player.

A genuine on-off metric requires both:

1. team performance while the player is on the court
2. team performance while the player is off the court

The current data does not provide a verified off-court comparison. Therefore net rating must not be labeled as on-off net rating.

## Team-impact family

The season-value model declares a team-impact family based on shrunk net rating and PIE fields.

For the two GOAT candidates, the produced FAMILY_TEAM_IMPACT column currently has no non-null values. The family therefore contributes no information and must not be described as active evidence.

Positive COVERAGE_TEAM_IMPACT values with null family scores are treated as a provenance or implementation mismatch requiring disclosure.

## Coverage asymmetry

Metric availability is era-dependent.

LeBron has broader NBA tracking-era coverage for net rating, PIE, and raw plus-minus. Jordan has historical Basketball Reference coverage for several box-derived impact metrics, but not equivalent modern tracking coverage.

Partial coverage is not interpreted as player weakness. Missing metrics receive no score and do not become zero.

## Release policy

For version 1:

- all audited impact metrics remain diagnostic
- central-score weight remains zero
- unavailable proprietary metrics remain unavailable
- net rating is not relabeled as on-off
- no missing value is imputed as player performance
- final simulation remains blocked until the full release freeze

This policy can be revisited only after a reproducible, era-comparable dataset is added and preregistered.
