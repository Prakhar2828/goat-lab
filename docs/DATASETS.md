# GOAT Lab v1 Data Inventory

## 1. Purpose

This document lists the data that entered the published GOAT Lab v1 pipeline or its released supporting views.

It is not a wishlist of possible future datasets.

No survey was conducted or used. No survey responses, external GOAT polls, or public-opinion variables entered the published v1 result.

## 2. Player-season data

Unit of analysis:

PLAYER_NAME + SEASON + SEASON_TYPE

Season types:

- Regular Season
- Playoffs

Fields used where available include:

- Games
- Minutes
- Points
- Field-goal attempts
- Free-throw attempts
- Assists
- Rebounds
- Steals
- Blocks
- Turnovers
- Per-100-possession rates
- Usage percentage
- Assist percentage
- Rebound percentage
- True-shooting percentage
- Offensive rating
- Defensive rating
- Net rating
- PIE

Main uses:

- Per-75 conversion
- True-shooting calculation
- Same-season league baselines
- Era-relative standardization
- Reliability adjustment
- Metric-family construction
- Season-value scoring
- Peak, prime, longevity, regular-season, playoff, offense, and defense categories

Unavailable historical fields are not entered as zero. Calculations use available components while coverage is tracked separately.

## 3. Historical league and career reference data

League reference values are calculated separately by season and season type.

These distributions support:

- League means
- League standard deviations
- Relative values
- Standardized z-scores
- Historical percentiles
- Reliability-adjusted metrics

Career reference eligibility requires:

- At least five regular seasons
- At least 5,000 regular-season minutes

The historical career reference population supplies distributions for:

- Peak
- Prime
- Longevity
- Regular-season value
- Playoff value
- Offense
- Historical box-score defense

The reference population prevents the category scale from being based only on the two target players.

## 4. Playoff-series data

Unit of analysis:

One team-side per playoff series

Required core fields include:

- SEASON
- TEAM_WON_SERIES

Available pre-series features are selected from:

- TEAM_SRS
- OPP_SRS
- TEAM_NET_RATING
- OPP_NET_RATING
- TEAM_SEED
- OPP_SEED
- HOME_COURT
- REST_ADVANTAGE
- TEAM_STAR_VALUE
- OPP_STAR_VALUE
- TEAM_SUPPORT_VALUE
- OPP_SUPPORT_VALUE

Derived fields include:

- EXPECTED_SERIES_WIN_PROB
- SERIES_OVERPERFORMANCE
- SURPRISE_LOG_SCORE
- CV_FOLD
- PREDICTION_SOURCE

Main uses:

- Training and evaluating the playoff-series expectation model
- Producing season-grouped out-of-fold predictions
- Building the winning-context category
- Supporting playoff and team-context dashboard views

The published model uses regularized logistic regression with median imputation and standard scaling.

Historical career scoring uses season-grouped out-of-fold predictions. A series is not scored by a model trained on another series from the same season.

## 5. Defensive evidence

The designed defense structure allows:

- Historical box-score defense
- Structured expert-film evidence
- Defensive-award evidence

Published v1 uses:

- Historical box-defense evidence
- Defensive-award evidence where available

No expert-film source met the frozen primary-model eligibility standard. No expert-film score entered the central v1 result.

Missing components are reweighted rather than treated as zero.

The following did not enter the released full-career defense score:

- Modern tracking data
- Possession-level lineup data
- RAPM
- Matchup-level tracking
- Film-possession annotations

## 6. Wikimedia pageviews

Unit of analysis:

PLAYER_NAME + DATE

Required fields include:

- PLAYER_NAME
- date
- views

Main use:

The common-window public-attention component of cultural impact.

The attention score uses:

- Total view share
- Median daily view share
- Median annual view share

Both players use the same overlapping comparison window.

Wikimedia attention contributes 20% of the published cultural-impact category.

## 7. Cultural-impact rubric

Unit of analysis:

PLAYER_NAME + DIMENSION

Required fields include:

- PLAYER_NAME
- DIMENSION
- SCORE_0_100
- CONFIDENCE
- SOURCE_IDS
- RATIONALE

Required dimensions:

- Commercial and global reach
- Basketball-culture influence
- Media and entertainment reach
- Philanthropy and social institutions

Dimension weights:

- Commercial and global reach: 30%
- Basketball-culture influence: 30%
- Media and entertainment reach: 15%
- Philanthropy and social institutions: 25%

A final rubric score is produced only when every required dimension is complete.

The sourced rubric contributes 80% of the published cultural-impact category.

The rubric is a declared evidence index. It is not a causal estimate of commercial, cultural, media, or social outcomes.

## 8. Defensive-award evidence

Defensive-award records are used as supporting historical evidence in the defense category.

Relevant evidence may include:

- Defensive Player of the Year recognition
- All-Defensive Team recognition
- Available voting or placement information

Award evidence is not treated as a complete measure of defense.

Awards can reflect:

- Reputation
- Voting context
- Positional expectations
- Team success
- Historical differences in award availability

## 9. Frozen category-score data

The released category-score artifacts contain the final 0–100 scores used by the published model.

The nine categories are:

- Peak
- Prime
- Longevity
- Regular season
- Playoffs
- Offense
- Defense
- Winning context
- Cultural impact

These artifacts support:

- The public result table
- Category comparisons
- Weight contribution views
- The custom weight simulator
- Release verification

## 10. Weight-simulation data

The frozen sensitivity release contains results from:

- 250,000 draws
- Random seed 23
- Within-group Dirichlet concentration 100
- Fixed group totals of 50% / 40% / 10%

Released outputs include:

- Player win shares
- Score summaries
- Margin summaries
- Category-weight correlations with the LeBron-minus-Jordan margin
- Driver tables used by the dashboard

The reported 60.1484% and 39.8516% values are shares of tested scoring setups. They are not objective GOAT probabilities.

## 11. Scaling-sensitivity data

The release evaluates four approved category-scaling methods:

- historical_percentile
- normal_score_tail
- bounded_logit_tail
- robust_mad_reference

The production method is bounded_logit_tail.

The four approved methods split 2–2 between LeBron and Jordan.

Scaling-sensitivity outputs are used to explain why the published conclusion is conditional rather than stable across every approved transformation.

## 12. Frozen release artifacts

The public dashboard reads committed release files covering:

- Production category scores
- Production hierarchy scores
- Player-season values
- Peak, prime, and longevity summaries
- Scored playoff-series data
- Playoff model metrics
- Weight-simulation summaries
- Weight-simulation drivers
- Scaling-sensitivity results
- Release-gate metadata
- Model training metadata
- Artifact hashes
- Machine-readable release manifest

These committed files allow the public dashboard to run without downloading source data or rerunning the final model and simulation.

## 13. Data not used in the published v1 result

The following did not enter the frozen v1 result:

- Survey responses
- External GOAT polling data
- RAPM
- Possession-level lineup regression
- Shot-chart data
- Play-type data
- Film-possession annotations
- Qualifying expert-film scores
- Google Trends
- GDELT
- New York Times article data
- PCA-derived category scores
- Bootstrap samples or confidence intervals
- Gradient-boosted playoff predictions

Some related code or early planning may remain in the research repository. Those components did not enter the published scores or conclusion.

## 14. Known data limitations

Important limitations include:

- Historical statistics are not equally complete across eras
- Advanced metrics are more available in recent seasons
- Rule changes and playing environments cannot be fully normalized
- Defensive evidence remains incomplete
- Playoff context is measured at the team-series level
- Supporting-cast and star-value fields are simplified summaries
- Wikimedia attention covers a modern overlapping period
- Cultural-rubric scoring contains structured human judgment
- Missing evidence can change the effective composition of a category

## 15. Release and versioning policy

The v1 release is frozen and reproducible.

A future data correction or methodology change should produce a new release rather than silently modifying the meaning of the published v1 result.

The public dashboard reads the committed v1 artifacts and does not rerun the final model or simulation.