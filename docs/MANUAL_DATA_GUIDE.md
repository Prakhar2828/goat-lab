# Manual data guide

## Why manual imports exist

Some high-value historical information is available through human-readable tables, filings, articles, video, or archives but lacks a stable public API. Manual imports make the project reproducible without silently depending on fragile scraping.

## Advanced player metrics

Create `data/manual/manual_advanced.csv` with:

```text
player,season,season_type,per,ts_pct,usg_pct,ows,dws,ws,ws_per_48,obpm,dbpm,bpm,vorp
```

Rules:

- Use canonical names `Michael Jordan` and `LeBron James`.
- Use NBA season strings such as `1990-91`.
- Use `Regular Season` or `Playoffs` exactly.
- Consolidate traded-team totals using the total row, not separate team rows.
- Preserve missing values as blank, never zero.
- Add the source table and access date to `sources.csv`.

## MVP voting

Create `data/manual/mvp_vote_shares.csv`:

```text
player,season,rank,first_place_votes,points_won,points_max,share
```

Use the published vote share. Voting systems changed historically, so raw vote counts should not be directly compared without the share.

## Playoff series

The model needs all playoff series, not only Jordan and LeBron series. Build one row per team-side. Each physical series therefore appears twice with opposite target values.

Recommended validation:

- Each `SERIES_ID` has exactly two team rows.
- Exactly one row has `TEAM_WON_SERIES = 1`.
- Team and opponent IDs reverse correctly.
- Pre-series features are identical mirror values where appropriate.

## Film annotations

Begin with a small calibration set. Both coders grade the same possessions, discuss disagreements, and revise rubric anchors before coding the final sample.

Do not select only famous highlights. Sample games and possessions using a documented random or stratified method.

## Impact ledger

Each row must represent one claim and one source. Examples:

- Verified donation amount
- Institution opened
- Published beneficiary count
- Annual brand revenue specifically attributable in a filing
- Documented player statement of influence
- Major media or league milestone

A source announcing an intention receives lower confidence than an audited outcome report.

## Google Trends

1. Open Google Trends.
2. Compare both exact topics in the same query.
3. Use worldwide, 2004-present, all categories, web search.
4. Export CSV.
5. Repeat for United States and selected global regions only if those comparisons are planned in advance.
6. Do not compare values exported from separate queries because each export has its own normalization.

## Sources registry

Every manual fact uses a `SOURCE_ID` linked to:

```text
source_id,title,publisher,url,published_date,accessed_date,source_type,primary_source,notes
```

Prefer official league data, filings, foundation reports, peer-reviewed research, direct interviews, and reputable reporting.

## Manual category evidence

`manual_category_inputs.csv` uses 0–100 evidence scores for contextual categories that lack a complete league-wide reference population:

- `winning_context_raw`: only when the playoff expectation model is unavailable or intentionally overridden
- `cultural_impact_raw`: synthesis of the four documented cultural subindexes
- `defense_film_score`: structured film score with coder agreement disclosed
- `defense_awards_score`: era-relative awards and voting evidence

The final defense category blends statistical, film, and awards evidence and reweights only across available components.
