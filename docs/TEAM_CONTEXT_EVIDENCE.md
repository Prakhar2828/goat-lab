# Team, Supporting-Cast, and Injury Context Policy

## Existing winning-context model

The playoff expectation model already uses team and opponent SRS and net rating in a season-grouped out-of-fold model. This controls for broad team strength before series overperformance is calculated.

Patch 7 does not replace that model or add result-dependent weights.

## Supporting-cast estimate

A diagnostic supporting-cast estimate is built from the existing league player season-value table.

For every Jordan and LeBron team-season:

1. duplicate measure-mode rows are collapsed by retaining the row with the largest minutes;
2. the focal player is excluded;
3. teammates are ranked by minutes;
4. the top eight teammates are retained;
5. available teammate season values are combined with minute weights;
6. missing teammate values reduce coverage and are never converted to zero.

This estimate is descriptive rather than causal. It does not isolate coaching, fit, role, opponent quality, or health.

## Injury and roster health

The repository does not contain a reproducible historical injury ledger with equivalent coverage for both careers. It also does not contain verified series-level top-eight availability for both teams.

Therefore:

- roster health remains missing;
- no injury score is fabricated;
- missing injury evidence does not become zero;
- supporting-cast context cannot enter the series expectation model in version 1;
- uncertainty must remain wider than it would be with complete health data.

## Coverage and era limitations

Modern NBA tables have better player and team coverage than early historical seasons. The diagnostic output reports coverage for every candidate season so the asymmetry is visible.

The estimate cannot be interpreted as a definitive answer to which player had more help. It is one structured view of teammate value under incomplete historical coverage.

## Double-counting control

Team strength is already represented by SRS and net rating in playoff expectation. Player value is already represented elsewhere in the category hierarchy.

Patch 7 therefore adds zero new central-score weight. Supporting-cast outputs are audit evidence only.

## Release rule

Supporting-cast or injury context may enter a future production model only after:

- both candidate and opponent context are available;
- health and availability definitions are preregistered;
- historical coverage is validated;
- leakage and double-counting tests pass;
- weights are frozen before viewing the final winner.

The final simulation remains blocked.
