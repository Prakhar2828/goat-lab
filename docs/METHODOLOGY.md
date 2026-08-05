# GOAT Lab v1 Methodology

## 1. Purpose

GOAT Lab compares Michael Jordan and LeBron James across nine declared categories:

1. Peak
2. Prime
3. Longevity
4. Regular-season value
5. Playoff value
6. Offense
7. Defense
8. Winning context
9. Cultural impact

The final score is a transparent multi-criteria comparison. It is not an objective GOAT detector and is not trained on public GOAT opinions.

No survey was conducted or used. No survey responses, external GOAT poll, or public-opinion variable entered any category or score.

## 2. Units of analysis

The main statistical unit is:

PLAYER_NAME + SEASON + SEASON_TYPE

Regular seasons and playoffs are kept separate.

A second dataset uses one team-side per playoff series for the winning-context model. Cultural evidence is stored separately and enters only the cultural-impact category.

The published v1 result does not use possession-level RAPM, lineup regression, shot-chart modeling, play-type modeling, or film-possession grades.

## 3. Data used in the published version

The executed v1 pipeline uses:

- Historical regular-season player statistics
- Historical playoff player statistics
- Traditional totals and rates
- Per-100-possession statistics where available
- Advanced statistics where available
- Games and minutes
- Playoff-series outcomes and pre-series context
- Defensive-award evidence
- Wikimedia pageviews over a shared comparison window
- A manually sourced cultural-impact rubric
- A historical career reference population
- Frozen release artifacts used by the public dashboard

Unavailable historical evidence is not entered as zero. Available components are used while coverage is tracked separately.

## 4. Pace adjustment

Per-100-possession statistics are converted to per-75 rates:

stat_per_75 = stat_per_100 × 0.75

This reduces pace differences between eras and represents an intuitive high-minute star workload.

It does not fully adjust for rule changes, spacing, offensive role, defensive schemes, or opponent quality.

## 5. Scoring efficiency

True shooting is calculated as:

TS% = PTS / [2 × (FGA + 0.44 × FTA)]

An existing valid true-shooting value is retained. The formula fills missing coverage when the required inputs exist.

## 6. Era-relative standardization

Every metric is compared with the league distribution from the same season and season type.

z = direction × (player value − league mean) / league standard deviation

League means and standard deviations are minute-weighted where possible.

Lower-is-better measures, such as defensive rating, are direction-adjusted so higher standardized values always represent better performance.

The feature layer can report:

- REL: difference from the same-season league mean
- Z: standard deviations above or below that mean
- PCTL: historical percentile
- Z_SHRUNK: reliability-adjusted z-score

## 7. Reliability adjustment

Small-minute samples are pulled toward league average:

reliability = minutes / (minutes + 500)

shrunk_z = z × reliability

This prevents a short playoff appearance from receiving the same confidence as a full season.

## 8. Transparent season-value model

The published season-value model groups related statistics into five interpretable families.

### Scoring

- Points per 75
- True-shooting percentage
- Usage percentage

### Playmaking

- Assists per 75
- Assist percentage

### Rebounding

- Rebounds per 75
- Rebound percentage

### Box-score defense

- Steals per 75
- Blocks per 75
- Direction-adjusted defensive rating

### Team impact

- Net rating
- PIE

Available reliability-adjusted metrics are averaged within each family. The available family scores are then averaged:

SEASON_VALUE_Z = mean(available family scores)

SEASON_VALUE_0_100 = clip(50 + 15 × SEASON_VALUE_Z, 0, 100)

A value near 50 is approximately league average on this display scale. It is not a percentage of basketball perfection.

PCA code exists as an experimental research utility, but no PCA-derived score entered the published v1 categories or final result.

## 9. Historical career reference population

Career values are compared with a wider historical player population rather than only comparing Jordan with LeBron.

A reference career must include:

- At least five regular seasons
- At least 5,000 regular-season minutes

This population supplies the historical career distributions used for peak, prime, longevity, regular season, playoffs, offense, and historical box-score defense.

## 10. Career category construction

### Peak

peak_raw = average of the best three regular seasons

### Prime

prime_raw = best seven consecutive regular seasons

Calendar gaps break consecutive windows.

### Longevity

longevity_raw = sum of max(season value − 50, 0)

Only season value above the approximate league-average threshold accumulates.

### Regular season

regular_season_raw = mean of the ten best regular seasons

### Playoffs

playoffs_raw = minutes-weighted mean of playoff season value

### Offense

offense_raw = minutes-weighted mean of scoring and playmaking

### Historical box defense

defense_raw = minutes-weighted mean of box-defense value

These career measures are compared with the historical reference population.

## 11. Defense category

The defense composite was designed to allow:

- Historical box-score defense: 50%
- Structured film score: 35%
- Defensive-award score: 15%

Missing components are reweighted rather than entered as zero.

For the published v1 release, no expert-film source met the frozen primary-model eligibility standard. No expert-film value entered the central score.

The available historical box and defensive-award evidence were therefore reweighted across the available components.

Modern tracking, lineup, possession, and RAPM metrics did not enter the full-career v1 defense score.

## 12. Playoff-series expectation model

Winning context uses regularized logistic regression with median imputation and feature standardization.

Available pre-series features are selected from:

- Team and opponent SRS
- Team and opponent net rating
- Team and opponent seed
- Home-court advantage
- Rest advantage
- Team and opponent star value
- Team and opponent supporting-cast value

Only features with usable values are included. At least four contextual features are required.

The prediction target is:

TEAM_WON_SERIES in {0, 1}

Career scoring uses season-grouped out-of-fold predictions with up to ten GroupKFold folds.

Every historical series is scored by a model that did not train on any series from that season. This prevents in-sample career scoring and keeps an entire season inside one fold.

For every team-side:

series_overperformance = actual series result − expected series win probability

The player-level winning-context score is:

winning_context = clip(50 + 50 × mean series overperformance, 0, 100)

This is team-level contextual evidence. It does not assign the entire series outcome to one player and is not a simple championship count.

The released model report includes ROC-AUC, log loss, Brier score, evaluation method, and fold count.

No gradient-boosted playoff model entered the published v1 result.

## 13. Cultural-impact category

Cultural impact combines:

- 20% common-window Wikimedia attention
- 80% sourced cultural rubric

### Wikimedia attention

The attention score combines:

- Total view share: 50%
- Median daily view share: 30%
- Median annual view share: 20%

Both players use the same overlapping comparison window.

### Sourced cultural rubric

The rubric contains:

- Commercial and global reach: 30%
- Basketball-culture influence: 30%
- Media and entertainment reach: 15%
- Philanthropy and social institutions: 25%

Every rubric row requires:

- A 0–100 score
- A confidence label
- Source identifiers
- A written rationale

A final rubric score is produced only when all required dimensions are complete.

The cultural score is a declared evidence index, not a causal estimate.

Google Trends, GDELT, New York Times data, and a public survey did not enter the frozen v1 cultural score.

## 14. Category scaling

Four approved category-scaling methods were evaluated:

1. historical_percentile
2. normal_score_tail
3. bounded_logit_tail
4. robust_mad_reference

The production method is bounded_logit_tail, frozen before the final simulation.

It is applied to:

- Peak
- Prime
- Longevity
- Regular season
- Playoffs
- Offense

Defense, winning context, and cultural impact remain on their native 0–100 evidence scales.

The four approved scenarios split 2–2 between LeBron and Jordan. The published conclusion is therefore conditional rather than stable across every approved scaling method.

## 15. Frozen hierarchy and weights

The three group totals are fixed:

- Performance Arc: 50%
- Basketball Value: 40%
- Broader Legacy: 10%

| Category | Weight |
|---|---:|
| Peak | 12.5% |
| Prime | 10.0% |
| Longevity | 7.5% |
| Regular season | 10.0% |
| Playoffs | 10.0% |
| Offense | 18.0% |
| Defense | 12.0% |
| Winning context | 10.0% |
| Cultural impact | 10.0% |

The final calculation is:

GOAT score = sum of category score × category weight

The weights were frozen before the final result was inspected.

## 16. Published result

| Player | Score | Rank |
|---|---:|---:|
| LeBron James | 89.258985 | 1 |
| Michael Jordan | 89.143895 | 2 |

Score difference:

89.258985 − 89.143895 = 0.115091

## 17. Weight stress test

The final stress test uses:

- 250,000 draws
- Random seed 23
- Within-group Dirichlet concentration 100
- Group totals fixed at 50% / 40% / 10%

Only relative category priorities inside multi-category groups vary.

Results:

- LeBron ranked first in 60.1484% of sampled setups
- Jordan ranked first in 39.8516%

These percentages describe the scoring setups that were tested. They are not objective probabilities that either player is the GOAT.

## 18. Evidence excluded from the central score

The following remained diagnostic and carried zero additional central weight:

- Expert-film consensus
- Supporting-cast audits beyond available pre-series model inputs
- Injury and roster-health gaps
- Locally available impact metrics
- Game-level playoff audits
- Cultural-weight sensitivity scenarios

The published v1 result also excludes:

- Surveys and public GOAT polls
- RAPM and possession-level lineup models
- Shot-chart and play-type models
- PCA-derived category scores
- Bootstrap confidence intervals
- Gradient-boosted playoff models

## 19. Reproducibility

The frozen release records:

- Source commit
- Production scaling method
- Exact category weights
- Random seed
- Number of simulations
- Dirichlet concentration
- Release-gate checks
- Artifact hashes
- Machine-readable manifest
- Committed dashboard data
- Automated tests and CI

The public dashboard reads committed release artifacts. It does not rerun the final model or simulation.

## 20. Interpretation and limitations

Appropriate language includes:

- Ranked first under this setup
- Predicted
- Exceeded model expectation
- Evidence supports
- Sensitive to assumptions

GOAT Lab does not prove an objective GOAT or assign complete team outcomes to one player.

The strongest supported conclusion is:

LeBron leads narrowly under the frozen production model, while the winner changes under other approved scaling choices and reasonable value priorities.