# Analytical data dictionary

## `league_player_seasons.parquet`

Raw/cached output from NBA season endpoints.

| Column | Meaning |
|---|---|
| PLAYER_ID | NBA player identifier |
| PLAYER_NAME | Canonical display name |
| SEASON | NBA season string |
| SEASON_TYPE | Regular Season or Playoffs |
| MEASURE_TYPE | Base or Advanced |
| PER_MODE | Totals or Per100Possessions |
| GP | Games played |
| MIN | Minutes |
| PTS/REB/AST/STL/BLK/TOV | Traditional production |
| OFF_RATING/DEF_RATING/NET_RATING | NBA advanced ratings when available |
| AST_PCT/REB_PCT/USG_PCT/PIE | NBA advanced shares/impact indicators |

## `league_player_features.parquet`

One analytical row per player-season-season type, after merging metric modes.

Derived columns:

| Column suffix | Meaning |
|---|---|
| `_PER100` | Production per 100 possessions |
| `_PER75` | Production per 75 possessions |
| `_REL` | Difference from season-specific league mean, direction adjusted |
| `_Z` | Season-specific standardized value |
| `_PCTL` | Season-specific percentile |
| `_Z_SHRUNK` | Z-score reduced toward average for small minutes |
| RELIABILITY | Minutes-based reliability weight |

## `league_player_season_values.parquet`

Adds:

| Column | Meaning |
|---|---|
| FAMILY_SCORING | Coverage-aware scoring family score |
| FAMILY_PLAYMAKING | Coverage-aware playmaking family score |
| FAMILY_REBOUNDING | Coverage-aware rebounding family score |
| FAMILY_DEFENSE_BOX | Coverage-aware box-score defense family score |
| FAMILY_TEAM_IMPACT | Coverage-aware team-impact family score |
| COVERAGE_* | Share of expected family metrics available |
| SEASON_VALUE_Z | Transparent mean of available family scores |
| SEASON_VALUE_0_100 | Display transformation, clipped to 0–100 |
| CAREER_YEAR | Active-career season index |

## `historical_career_reference.parquet`

Career summaries for players with at least five regular seasons and 5,000 minutes. This is the calibration population for target-player category percentiles.

## `category_scores.parquet`

| Column | Meaning |
|---|---|
| peak | Historical percentile of top-three-season peak |
| prime | Historical percentile of best seven-year consecutive prime |
| longevity | Historical percentile of accumulated value above average |
| regular_season | Historical percentile of top-ten regular seasons |
| playoffs | Historical percentile of minutes-weighted playoff season value |
| offense | Historical percentile of scoring/playmaking evidence |
| defense | Blend of historical statistical percentile, structured film, and awards evidence |
| winning_context | 0–100 contextual series-overperformance evidence |
| cultural_impact | 0–100 synthesis of separately documented cultural subindexes |

## `playoff_series_scored.parquet`

| Column | Meaning |
|---|---|
| EXPECTED_SERIES_WIN_PROB | Pre-series model probability |
| SERIES_OVERPERFORMANCE | Actual binary result minus expected probability |
| SURPRISE_LOG_SCORE | Negative log probability of observed result |

## `weight_simulation_summary.parquet`

| Column | Meaning |
|---|---|
| WIN_RATE | Share of sampled value systems won |
| MEAN_SCORE | Average weighted score across simulations |
| P05_SCORE/P95_SCORE | Weight-system score interval |
