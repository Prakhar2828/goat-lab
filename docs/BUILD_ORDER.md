# Complete build order

This order minimizes rework. Do not begin styling the final dashboard before the metric registry, data coverage, and scoring rules are frozen.

## Stage 0 — Freeze the research contract

Write these statements at the top of the methodology:

1. The project compares three related but distinct outcomes: best peak, greatest career, and greatest total basketball figure.
2. No single metric is treated as ground truth.
3. Missing era-specific data is not treated as poor performance.
4. All category weights are visible and adjustable.
5. Every result has a comparison window and evidence-confidence level.
6. The public release is frozen to a named season cutoff.

## Stage 1 — Repository and environment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
pytest -q
```

Create a GitHub repository immediately and commit the empty architecture before adding data.

## Stage 2 — Core NBA ingestion

Run first:

```bash
goatlab ingest-core --skip-game-logs
```

Inspect:

```text
data/interim/league_player_seasons.parquet
data/interim/league_team_seasons.parquet
data/interim/player_awards.parquet
```

Then run the full game-log ingestion. Check row counts by season and season type. Save failed request details rather than dropping a season.

## Stage 3 — Advanced and award data

Export and normalize:

- Jordan regular-season advanced table
- Jordan playoff advanced table
- LeBron regular-season advanced table
- LeBron playoff advanced table
- MVP voting shares for every season in which either received votes

Do not manually calculate BPM, VORP, PER, or Win Shares from incomplete formulas. Import the published values and cite the methodology source.

## Stage 4 — Data validation

Run the following assertions:

- Player IDs map to exactly one canonical player.
- Player-season-team rows are consolidated correctly when a player changed teams.
- `PTS = 2*FG2M + 3*FG3M + FTM` within expected data tolerance.
- Shooting percentages are inside `[0, 1]`.
- Minutes and games are nonnegative.
- Regular-season and playoff rows are never mixed.
- Championship and award counts match independent references.
- No duplicated player-season-season-type rows in the analytical table.
- Each metric’s earliest and latest season match the registry.

## Stage 5 — Feature engineering

Build, in this order:

1. Per-75 and per-100 production
2. True shooting and relative true shooting
3. League-relative z-scores and percentiles
4. Sample-size reliability shrinkage
5. Metric-family scores
6. Transparent season value
7. PCA latent season value as a robustness model
8. Career year and age curves
9. Peak and consecutive-prime windows
10. Cumulative value above replacement, average, All-Star, and All-NBA thresholds

Inspect every transformation using at least three non-target players. A metric can look reasonable for Jordan and LeBron while being wrong league-wide.

## Stage 6 — Team and playoff context

Create one row for each team-side of every playoff series. Required pre-series features:

- Team and opponent SRS
- Team and opponent net rating
- Seed
- Home-court advantage
- Rest difference
- Team’s highest player-value score
- Supporting-cast value excluding the star
- Availability of top rotation players
- Round and best-of format

The target is whether that team won the series. Split train/test by complete seasons, never random rows, because both sides of a series are dependent and basketball environments change over time.

Output:

- Expected series-win probability
- Actual minus expected series outcome
- Aggregate playoff-path difficulty
- Opponent SRS defeated
- Upset credit
- Favorite-loss penalty

Do not use Finals record as an isolated metric.

## Stage 7 — Defense and film coding

Create a stratified game sample for each player:

- Peak regular season
- Peak postseason
- Championship run
- Elimination game
- Strong offensive game
- Weak offensive game
- Young-career game
- Late-career game

For each defensive possession, annotate:

- Primary assignment
- Point-of-attack containment
- Screen navigation
- Help rotation
- Rim deterrence
- Defensive rebounding responsibility
- Transition effort
- Communication or visible organization
- Error severity
- Matchup difficulty

Use at least two coders for a serious final claim. Report Cohen’s kappa or an intraclass correlation coefficient. If only one coder is available, label the output “structured film assessment,” not objective defensive truth.

## Stage 8 — Cultural, commercial, and social evidence

Keep four separate indexes:

1. Public attention
2. Commercial influence
3. Social/philanthropic impact
4. Influence on players and basketball culture

Do not add raw dollars, pageviews, beneficiary counts, and interview mentions directly. Normalize within each evidence type, document inflation adjustments, and report overlapping-period versus historical-legacy evidence separately.

## Stage 9 — Argument explorer

Every argument must contain:

- Strongest fair formulation
- Player it usually supports
- Supporting evidence
- Strongest counterargument
- Relevant metrics
- Data limitations
- Current verdict
- Confidence
- Source IDs

Avoid straw-man arguments. The goal is to make both sides feel represented accurately.

## Stage 10 — Final category scores

Publish both:

- Raw evidence summaries
- 0–100 display scores

Display scores should be calibrated against a broader historical reference population where possible. Do not min-max only Jordan and LeBron in the final release because that exaggerates small differences.

## Stage 11 — Sensitivity and uncertainty

Run:

- 250,000+ Dirichlet weight samples
- Bootstrap season resampling
- Leave-one-metric-out tests
- Leave-one-family-out tests
- Modern-data-only exclusion tests
- Equal-weight metric-family model
- PCA robustness model
- Alternative peak windows
- Alternative replacement and elite thresholds

The final conclusion should report which assumptions are required for each player to lead.

## Stage 12 — Publication build

Before sharing:

- Freeze processed data
- Save source registry and hashes
- Save model reports
- Add methodology and limitations pages
- Add downloadable CSV of category evidence
- Test on mobile width
- Test every slider and filter
- Remove unsupported causal language
- Add a concise executive conclusion
- Tag a GitHub release
