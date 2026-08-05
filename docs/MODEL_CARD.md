# GOAT Lab v1 Model Card

## 1. Model overview

GOAT Lab is a transparent decision-support project that compares Michael Jordan and LeBron James across multiple definitions of basketball greatness.

The project combines nine declared categories:

- Peak
- Prime
- Longevity
- Regular-season value
- Playoff value
- Offense
- Defense
- Winning context
- Cultural impact

GOAT Lab does not claim to identify one objective GOAT. Its output depends on declared category definitions, scaling choices, evidence coverage, and value weights.

No survey was conducted or used. No survey response, external GOAT poll, or public-opinion variable entered the published v1 result.

## 2. Intended uses

GOAT Lab is intended for:

- Exploring how different definitions of greatness affect the Jordan–LeBron comparison
- Demonstrating transparent sports-analytics methodology
- Comparing career performance across eras
- Examining the effect of category weights and scaling assumptions
- Supporting informed basketball discussion
- Demonstrating a reproducible public data-science project

## 3. Uses outside the project’s scope

GOAT Lab is not intended to:

- Prove an objective GOAT
- Support sports betting
- Predict future NBA games
- Evaluate employment or contract decisions
- Replace scouting or expert film analysis
- Assign complete team outcomes to one player
- Make causal claims about culture, sales, globalization, philanthropy, or team success

## 4. Published model components

### 4.1 Transparent season-value model

Purpose:

Summarize each player-season using interpretable statistical families rather than one opaque metric.

Input families:

- Scoring
- Playmaking
- Rebounding
- Box-score defense
- Team impact

Metrics are adjusted relative to the same season and season type. Small-minute samples are reduced using reliability shrinkage.

The available family scores are averaged into a season-value z-score.

The display score is:

SEASON_VALUE_0_100 = clip(50 + 15 × SEASON_VALUE_Z, 0, 100)

A value around 50 is approximately league average on this display scale. It is not a percentage of basketball perfection.

Main risks:

- Historical metric coverage differs by era
- Box-score families do not capture every basketball contribution
- Family construction remains a modeling choice
- Equal averaging of available families is not the only defensible structure

### 4.2 Historical career-category model

Purpose:

Convert season-level evidence into career-level measures.

Published career definitions:

- Peak: average of the best three regular seasons
- Prime: best seven consecutive regular seasons
- Longevity: accumulated season value above the approximate league-average threshold
- Regular season: mean of the ten best regular seasons
- Playoffs: minutes-weighted mean of playoff season value
- Offense: minutes-weighted scoring and playmaking value
- Historical box defense: minutes-weighted box-defense value

Historical reference eligibility:

- At least five regular seasons
- At least 5,000 regular-season minutes

Main risks:

- Career-window definitions represent declared value choices
- Historical reference coverage is not identical across all eras
- Longevity rewards both quality and duration
- Best-season and consecutive-window definitions can favor different career shapes

### 4.3 Category-scaling model

Purpose:

Convert historical career measures to comparable category scales.

Approved methods:

- historical_percentile
- normal_score_tail
- bounded_logit_tail
- robust_mad_reference

The frozen production method is bounded_logit_tail.

It is applied to:

- Peak
- Prime
- Longevity
- Regular season
- Playoffs
- Offense

Defense, winning context, and cultural impact remain on their native 0–100 evidence scales.

Main risk:

The winner changes across approved scaling methods. The four approved scenarios split 2–2 between Jordan and LeBron.

This is a central reason the published conclusion is described as conditional rather than universally robust.

### 4.4 Defense evidence model

Designed structure:

- Historical box-score defense: 50%
- Structured expert film: 35%
- Defensive awards: 15%

Missing components are reweighted rather than treated as zero.

Published v1 treatment:

No expert-film source met the frozen primary-model eligibility standard. No expert-film score entered the central result.

The available historical box-defense and defensive-award evidence were reweighted across the available components.

Main risks:

- Historical defense is difficult to measure consistently
- Box statistics represent only part of defensive value
- Awards reflect reputation, voting context, and positional expectations
- Comparable modern tracking evidence does not exist across both careers
- No qualifying structured film value entered v1

### 4.5 Playoff-series expectation model

Purpose:

Estimate a team’s expected probability of winning a playoff series using information available before the series.

Model:

- LogisticRegression
- Regularization parameter C = 0.5
- Median imputation
- Standard feature scaling
- Random seed 23

Available feature groups include:

- Team and opponent SRS
- Team and opponent net rating
- Team and opponent seed
- Home-court advantage
- Rest advantage
- Team and opponent star value
- Team and opponent supporting-cast value

Evaluation method:

Historical career scoring uses season-grouped out-of-fold predictions with up to ten GroupKFold folds.

A historical series is scored by a model that did not train on any series from the same season.

Released evaluation metrics:

- ROC-AUC: 0.820393
- Log loss: 0.518479
- Brier score: 0.173115

Derived value:

series_overperformance = actual series result − expected series win probability

Winning-context score:

winning_context = clip(50 + 50 × mean series overperformance, 0, 100)

Interpretation:

This is team-level contextual evidence. It does not prove that the focal player caused the result or deserves all credit or blame.

Main risks:

- Historical pre-series context is incomplete
- Feature availability varies by season
- Team-level residuals cannot isolate individual causation
- Logistic regression may not capture every nonlinear relationship
- Supporting-cast and star-value inputs remain simplified summaries

### 4.6 Cultural-impact model

Purpose:

Represent broader influence beyond the court using declared evidence components.

Published structure:

- Common-window Wikimedia attention: 20%
- Manually sourced cultural rubric: 80%

Wikimedia attention combines:

- Total view share: 50%
- Median daily view share: 30%
- Median annual view share: 20%

The sourced rubric combines:

- Commercial and global reach: 30%
- Basketball-culture influence: 30%
- Media and entertainment reach: 15%
- Philanthropy and social institutions: 25%

Every rubric entry requires:

- A 0–100 score
- A confidence label
- Source identifiers
- A written rationale

Main risks:

- Cultural evidence is less standardized than basketball statistics
- Digital attention data covers a modern overlapping period
- Rubric scoring requires judgment
- Attention does not necessarily represent positive influence
- The category is descriptive rather than causal

Google Trends, GDELT, New York Times data, and public survey data did not enter the frozen v1 cultural score.

### 4.7 Final weighted comparison

Purpose:

Combine the nine category scores using weights frozen before the final result was inspected.

Frozen weights:

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

Fixed group totals:

- Performance Arc: 50%
- Basketball Value: 40%
- Broader Legacy: 10%

Published production result:

| Player | Score | Rank |
|---|---:|---:|
| LeBron James | 89.258985 | 1 |
| Michael Jordan | 89.143895 | 2 |

The central margin is 0.115091 points on the 100-point display scale.

Main risk:

There is no value-neutral set of GOAT category weights.

### 4.8 Hierarchy-aware weight stress test

Purpose:

Measure how often each player leads when category priorities vary within the frozen major-group totals.

Configuration:

- 250,000 draws
- Random seed 23
- Dirichlet concentration 100
- Performance Arc fixed at 50%
- Basketball Value fixed at 40%
- Broader Legacy fixed at 10%

Results:

- LeBron ranked first in 60.1484% of sampled setups
- Jordan ranked first in 39.8516% of sampled setups

Interpretation:

These percentages represent shares of the tested scoring setups.

They are not probabilities that either player objectively is the GOAT.

Main risks:

- The stress test explores only the declared hierarchy
- Major-group totals remain fixed
- The Dirichlet concentration controls how far weights move from the central setup
- Other defensible GOAT frameworks could use different categories or group totals

## 5. Components not used in the published v1 result

The frozen v1 result does not include:

- Survey responses
- External GOAT polls
- RAPM
- Possession-level lineup regression
- Shot-chart modeling
- Play-type modeling
- Film-possession grades
- PCA-derived category scores
- Bootstrap confidence intervals
- Gradient-boosted playoff models
- Qualifying expert-film scores
- Google Trends
- GDELT
- New York Times article data

Some related experimental code or early planning may exist in the research repository. Those components did not enter the published category scores or final conclusion.

## 6. Data limitations

Important limitations include:

- Historical statistics are not equally complete across eras
- Advanced metrics are more available for recent seasons
- Rule changes and playing environments cannot be fully normalized
- Team context is difficult to separate from individual performance
- Defensive evidence remains especially incomplete
- Cultural evidence contains structured judgment
- Historical source corrections may change future versions
- Missing evidence is reweighted, which changes the effective composition of some categories

## 7. Fairness and comparability considerations

The project attempts to improve cross-era comparability through:

- Same-season league baselines
- Pace adjustment
- Reliability shrinkage
- Historical reference distributions
- Explicit missing-data handling
- Shared cultural-attention windows
- Transparent category construction
- Scaling sensitivity analysis

These procedures reduce some era-related distortions but do not eliminate them.

## 8. Reproducibility and release integrity

The published release records:

- Frozen source commit
- Production scaling method
- Exact category weights
- Random seed
- Number of simulations
- Dirichlet concentration
- Release-gate checks
- Artifact hashes
- Machine-readable manifest
- Committed dashboard data
- Automated tests
- Continuous integration results

The public dashboard reads committed release artifacts. It does not rerun the final model or simulation.

## 9. Monitoring and future revisions

A future version should receive a new release rather than silently modifying the frozen v1 result.

Possible future improvements include:

- Stronger historical data validation
- Better cross-era defensive evidence
- Independently reviewed cultural-rubric scoring
- Additional sensitivity structures
- More complete pre-series team context
- Qualified structured film evidence
- Formal uncertainty intervals at the correct clustering level

These are future extensions and should not be described as part of the published v1 model.

## 10. Required interpretation

The strongest supported conclusion is:

LeBron leads narrowly under the frozen production model, while Jordan leads under other approved scaling choices and reasonable alternative value priorities.

The model supports an assumption-sensitive comparison. It does not prove one universal answer to the GOAT debate.