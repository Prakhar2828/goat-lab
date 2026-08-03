# Dataset inventory

## A. Essential on-court datasets

| Dataset | Unit | Required fields | Main use | Coverage rule |
|---|---|---|---|---|
| Player season traditional | Player-season-season type | GP, MIN, PTS, FGA, FTA, 3PA, REB, AST, STL, BLK, TOV | Production, availability, efficiency | Entire target careers |
| Player season per 100 | Player-season-season type | Per-100 PTS, AST, REB, STL, BLK, TOV | Pace-neutral comparison | Entire target careers where available |
| Player season advanced | Player-season-season type | ORtg, DRtg, NetRtg, AST%, REB%, USG%, TS%, PIE | Role and team-impact context | Record exact start year per metric |
| Basketball-Reference advanced | Player-season-season type | PER, WS, WS/48, OBPM, DBPM, BPM, VORP | Independent metric family | BPM/VORP available for both target careers |
| Team season | Team-season-season type | W, L, pace, ORtg, DRtg, NetRtg | Era and team strength | Entire target careers |
| Team game logs | Team-game | Date, opponent, points, result | SRS, schedule strength, playoff series | Entire target careers |
| Player game logs | Player-game | Minutes and box score | Distribution, consistency, elimination/clutch samples | Entire target careers where endpoint succeeds |
| Awards | Player-award-season | Award, team, season | Accolades and defensive recognition | Entire careers |
| MVP voting | Player-season | Rank, votes, points, share | Voting dominance, not only award wins | Every vote-receiving season |

## B. Context datasets

### Playoff series table

One row per team-side per series. Minimum columns:

```text
SEASON, ROUND, SERIES_ID, TEAM_ID, OPP_TEAM_ID, TEAM_WON_SERIES,
TEAM_SEED, OPP_SEED, HOME_COURT, TEAM_SRS, OPP_SRS,
TEAM_NET_RATING, OPP_NET_RATING, REST_ADVANTAGE,
TEAM_STAR_VALUE, OPP_STAR_VALUE, TEAM_SUPPORT_VALUE, OPP_SUPPORT_VALUE
```

Optional but valuable:

```text
TEAM_TOP8_MINUTES_AVAILABLE, OPP_TOP8_MINUTES_AVAILABLE,
COACH_TENURE, PRESEASON_TITLE_PROB, SERIES_START_INJURY_NOTES,
BEST_OF, GAMES_PLAYED, PLAYER_NAME, PLAYER_SERIES_VALUE
```

### Roster and teammate value

Create a player-team-season table with minutes and season value. Supporting-cast value should exclude the focal player. Calculate several versions:

- Minutes-weighted mean value of teammates
- Top-three teammate value
- Top-eight rotation value
- Replacement-level minutes
- All-Star/All-NBA teammate indicators
- Team performance without the focal player

Do not use only named-star counts.

### Injuries and availability

There is no perfectly standardized free historical injury dataset. Use a manual series-level ledger with source citations. At minimum record whether a top-three rotation player was unavailable or meaningfully limited at series start.

## C. Possession and lineup datasets

### `shufinskiy/nba_data`

Repository: `https://github.com/shufinskiy/nba_data`

Uses NBA.com, data.nba.com, and pbpstats-derived data. It provides play-by-play beginning in 1996-97 and playoff data. Use it for:

- Possession parsing
- Lineups on court
- On/off values
- RAPM or regularized lineup models
- Shot details
- Matchups in modern seasons

Fairness rule: possession and lineup metrics cannot represent Jordan’s full career. Use them as matched-window or supplementary robustness evidence, not as an unqualified career category.

### `pbpstats`

Documentation: `https://pbpstats.readthedocs.io/`

Useful enriched concepts include possession start type, lineup IDs, score margin, event timing, and possession details.

## D. Shot and play-type datasets

### Shot charts

Use NBA shot-chart endpoints where available. Derive:

- Rim, short midrange, long midrange, corner three, above-break three frequencies
- Efficiency by zone
- Assisted versus unassisted share where available
- Playoff change in shot profile

Do not claim full-career shot-location equivalence when early data is missing.

### Synergy-style play types

Public repository: `https://github.com/DomSamangy/NBA_Play_Types_12_25`

Coverage begins in 2012-13, so it is a LeBron-era role analysis, not a direct full-career Jordan comparison. Use it to explain later-career offensive versatility and explicitly label the missing Jordan comparison.

## E. Cultural and public-impact datasets

### Wikimedia Analytics API

Documentation: `https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/`

Daily pageviews begin July 1, 2015. Use monthly totals, event-normalized spikes, geographic data where available, and overlapping-period comparisons.

### Google Trends

Manually export searches for the matched topics `Michael Jordan` and `LeBron James`. Use the same geography, category, search type, and date range. Because Trends values are normalized within each request, both names must appear in the same request for direct comparison.

### GDELT DOC API

Use for news-volume and tone analysis in overlapping modern years. Search exact names and remove common false positives. Validate a sample of articles manually.

### New York Times Article Search API

Optional single-publication historical corpus. It is not global media coverage, but its long archive can support a clearly labeled case study.

### Verified impact ledger

Manually create one row per sourced event:

```text
PLAYER_NAME, DATE, DIMENSION, SUBDIMENSION, EVENT,
RAW_VALUE, UNIT, INFLATION_ADJUSTED_USD, BENEFICIARIES,
GEOGRAPHY, SOURCE_ID, CONFIDENCE, NOTES
```

Dimensions:

- Commercial
- Philanthropic
- Educational
- Civic
- Basketball-culture influence
- Globalization
- Athlete-business influence

Never treat announcements as verified outcomes without evidence.

## F. Film annotation dataset

Minimum columns:

```text
PLAYER_NAME, GAME_ID, DATE, SEASON, SEASON_TYPE, ROUND,
POSSESSION_ID, PERIOD, CLOCK, OFFENSE_TEAM, DEFENSE_TEAM,
PRIMARY_ASSIGNMENT, ACTION_TYPE, OUTCOME,
POA_GRADE, SCREEN_NAV_GRADE, HELP_GRADE, RIM_GRADE,
REBOUND_GRADE, TRANSITION_GRADE, ERROR_SEVERITY,
MATCHUP_DIFFICULTY, CODER_ID, NOTES, VIDEO_SOURCE_ID
```

Use ordinal grades with written anchors, for example `-2` major harmful error through `+2` major positive impact.

## G. Survey dataset

A public survey cannot be completed rigorously in one day unless respondents already exist. Build the form now and publish the first dashboard version without presenting an empty or convenience sample as population truth.

Suggested fields:

- GOAT choice
- Peak importance
- Longevity importance
- Championship importance
- Statistics importance
- Defense importance
- Cultural-impact importance
- Age range
- Country/region
- Years watching NBA
- Watched Jordan live
- Watched LeBron live
- Playing/coaching experience

Analyze choice using logistic regression only after adequate sample size and demographic disclosure.
