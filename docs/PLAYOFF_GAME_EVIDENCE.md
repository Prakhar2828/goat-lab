# Playoff Game Evidence

## Scope

This patch adds a complete game-level playoff audit for Michael Jordan and LeBron James using the local historical player box-score dataset.

The audit matches every target player-game to the existing candidate series table using:

- player name
- team ID
- opponent team ID
- series start and end dates
- exact series game count

The release gate requires all 94 candidate series to match exactly.

## Available evidence

The source contains complete core box-score coverage for:

- 179 Michael Jordan playoff games
- 302 LeBron James playoff games
- 481 total candidate games
- 37 Jordan series
- 57 LeBron series

The core audit uses minutes, points, assists, rebounds, steals, blocks, turnovers, field goals, and free throws.

## Derived metrics

### True shooting percentage

True shooting percentage is calculated as:

`PTS / (2 * (FGA + 0.44 * FTA))`

The audit also reports a playoff-season-relative true-shooting index. A value of 100 represents the aggregate true-shooting percentage of qualifying player-games in that same postseason.

### Game Score

The audit calculates the transparent Hollinger Game Score box-score composite. It is useful as a compact descriptive statistic, but it is not treated as a causal impact estimate.

### Season-relative Game Score

Game Score and per-36 box metrics are standardized within the same playoff season. Player-games under 12 minutes are excluded from baseline estimation to reduce extreme low-minute rate noise.

### Elimination and closeout games

Series state is reconstructed before each game. The audit identifies:

- games in which the player's team faced elimination
- games in which the player's team could close the series
- Game 7s
- series-clinching wins

These splits are descriptive and are not assigned extra narrative weight.

## Uncertainty

Career comparisons are bootstrapped by playoff series rather than by individual game.

This preserves some within-series dependence and avoids pretending that seven games from one matchup are seven fully independent experiments.

The resulting intervals are descriptive bootstrap intervals, not proof that one player is universally superior.

## Central-score policy

All game-level outputs are diagnostic in this patch.

Additional central-score weight is fixed at zero because game-level box metrics overlap with the existing playoff, offense, defense, peak, and prime evidence. Adding them directly would risk double counting.

The final preregistration gate may test game-level evidence as a sensitivity scenario, but the result cannot be used to choose weights after seeing which player benefits.

## Limitations

Game-level box scores do not fully measure:

- defensive positioning and deterrence
- off-ball offensive value
- spacing and screening
- opponent game-plan quality
- teammate availability
- coaching and scheme
- possession-level clutch value
- causal on/off impact

Player plus-minus is retained only as a team-dependent diagnostic and is not required for the core evidence gate.

## Release rule

Patch 8 passes only when:

- all 481 candidate games are matched
- all 94 series have exact game counts
- core-stat coverage is complete
- no duplicate player-games exist
- no game-level metric changes the central score
- final simulation remains blocked
