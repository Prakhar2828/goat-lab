# Methodology

## 1. Research questions

GOAT Lab answers four separate questions before any combined score:

1. Who reached the highest basketball peak?
2. Who sustained elite play for the strongest prime?
3. Who created the most total career basketball value?
4. Who had the greatest broader impact on basketball and society?

The combined result is a weighted synthesis, not an objective law of basketball.

## 2. Unit of analysis

The primary unit is a player-season-season type. Regular season and playoffs remain separate. Game, series, possession, lineup, film-possession, and impact-event tables are linked downstream.

## 3. Pace adjustment

Per-100-possession rates are converted to per-75 rates for an intuitive high-minute star workload:

```text
stat_per_75 = stat_per_100 × 0.75
```

This controls for team pace but does not fully control for role, spacing, rules, or opponent quality.

## 4. Relative efficiency

True shooting:

```text
TS% = PTS / [2 × (FGA + 0.44 × FTA)]
```

Relative true shooting:

```text
rTS% = player TS% − league TS%
```

The league baseline is season and season-type specific. Minute-weighted league means are used where appropriate.

## 5. Era standardization

For metric `m` in season `s`:

```text
z(player,m,s) = direction(m) × [player(m,s) − league_mean(m,s)] / league_sd(m,s)
```

Lower defensive rating is direction-adjusted so higher standardized values are always better. Percentiles are also reported because z-scores are sensitive to distribution shape.

## 6. Sample-size shrinkage

Small playoff samples are shrunk toward league average:

```text
reliability = minutes / (minutes + prior_minutes)
shrunk_z = z × reliability
```

The prior-minutes constant is documented and varied in sensitivity analysis.

## 7. Metric families and double counting

Metrics are grouped into families such as scoring, playmaking, defense, rebounding, team impact, and availability. Highly correlated metrics are not all given independent full weight.

Two season-value models are published:

1. Transparent equal/fixed-weight family average
2. PCA latent factor robustness model

A conclusion that exists only under one correlated metric construction is labeled unstable.

## 8. Coverage-aware scoring

Every metric has:

- Earliest season
- Latest season
- Source
- Unit
- Direction
- Metric family
- Eligible comparison windows
- Missingness policy

When a metric is unavailable for one player because it did not exist, it is excluded and remaining weights are renormalized within the family. It is never imputed as zero.

Three result modes are required:

1. Full available evidence
2. Matched-data-only evidence
3. Modern-only metrics removed

## 9. Peak, prime, and longevity

Reported peak measures:

- Best season
- Top three seasons regardless of sequence
- Best three consecutive seasons
- Best five consecutive seasons
- Best seven consecutive seasons
- Top ten seasons

Longevity measures:

```text
career_value_above_threshold = Σ max(season_value − threshold, 0)
```

Thresholds are defined for average, All-Star, and All-NBA-level seasons and tested under alternatives.

## 10. Postseason model

The playoff model predicts the probability that a team wins a series using only information available before the series begins.

Suggested model hierarchy:

1. Logistic regression baseline
2. Gradient-boosted trees robustness model
3. Calibrated probability comparison

Features include SRS, net rating, seed, home court, rest, star value, supporting-cast value, and availability.

Validation uses chronological season splits. Random row splitting is prohibited because it leaks era information and can place two sides of the same series across train and test.

Series overperformance:

```text
series_overperformance = actual_result − expected_win_probability
```

This does not prove that the focal player alone caused the result. It is contextual evidence.

## 11. Teammate quality

Supporting cast excludes the focal player. Multiple definitions are shown:

- Top-three teammate quality
- Rotation-weighted quality
- Total teammate value
- Replacement-level minutes
- All-Star/All-NBA indicators

No single teammate model is treated as definitive.

## 12. Defense

Defense combines:

- Box-score indicators
- Team/on-off indicators where available
- Awards and voting
- Matchup/role information
- Structured film grades

Modern tracking data is supplementary and is not allowed to decide a full-career comparison by itself.

## 13. Cultural-impact index

The broader-impact section contains separate subindexes rather than one undifferentiated number:

- Public attention
- Commercial influence
- Social/philanthropic outcomes
- Influence on players and basketball culture

Digital data is separated into overlapping-period analysis and historical legacy evidence. Raw media attention is not interpreted as positive impact without context.

## 14. Combined GOAT score

For player `p`:

```text
GOAT(p) = Σ weight(category) × score(p,category)
```

Weights are nonnegative and sum to one. Missing categories trigger visible warnings and reweighting only when the user permits it.

## 15. Sensitivity analysis

Random weight vectors are sampled from a Dirichlet distribution. Report:

- Share of definitions won by each player
- Score intervals
- Categories most correlated with the margin
- Decision boundaries between major values
- Profiles under which the conclusion reverses

Run at least 250,000 simulations for the publication build.

## 16. Uncertainty

Use bootstrap resampling of seasons, games, series, and film possessions at the correct clustering level. Do not bootstrap individual possessions as independent when they are nested within games.

Report uncertainty for:

- Season-value estimates
- Peak windows
- Playoff model probabilities
- Film grades
- Cultural attention summaries
- Monte Carlo result shares

## 17. Causal-language rules

Allowed:

- Associated with
- Coincided with
- Predicted
- Outperformed model expectation
- Evidence supports

Avoid without a causal design:

- Caused league globalization
- Single-handedly won
- Proved a player made teammates better
- Directly generated all brand revenue

## 18. Final conclusion format

The conclusion must include:

1. Peak leader
2. Prime leader
3. Career-value leader
4. Postseason-context leader
5. Broader-impact leader
6. Balanced-profile leader
7. Percentage of defensible weight systems won
8. Largest remaining uncertainty
9. Conditions that reverse the result
