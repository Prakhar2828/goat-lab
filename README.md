# GOAT Lab — Michael Jordan vs. LeBron James

GOAT Lab is a reproducible basketball-research platform that asks a more precise question than “Who is the GOAT?”:

> Under which defensible definitions of basketball greatness does Michael Jordan or LeBron James rank first, and how robust is that conclusion?

The platform separates measurable basketball performance from value judgments. It analyzes peak, prime, longevity, regular-season value, postseason performance, opponent and teammate context, offense, defense, cultural influence, commercial impact, philanthropy, and argument-level evidence. It then lets a reader change category weights and see why the result changes.

## What is included

- Cached NBA data ingestion with retries and request throttling
- Regular-season and playoff player-season tables
- Per-75/per-100 possession normalization
- League-relative efficiency, z-scores, percentiles, and sample-size shrinkage
- Coverage-aware metric families so unavailable historical data does not count as zero
- Peak, consecutive-prime, top-ten-season, and cumulative-career models
- Team SRS estimation from game logs
- Playoff-series expectation model with temporal validation
- Series overperformance and surprise measures
- Monte Carlo sensitivity analysis over 250,000 weight combinations
- Wikimedia attention ingestion
- Manual source ledgers for business, philanthropy, influence, injuries, and film coding
- Multi-page Streamlit dashboard
- Methodology, data dictionary, model card, tests, and publication checklist

## Important research principle

This is not primarily a supervised-learning classification problem. A model trained to output “Jordan” or “LeBron” would learn historical opinions or arbitrary labels. The overall conclusion is a multi-criteria decision analysis. Machine-learning models are used only where a prediction target is meaningful, such as expected playoff-series outcomes or latent correlated performance factors.

## Stack

- Python 3.11+
- pandas / Polars / DuckDB / Parquet
- scikit-learn / statsmodels / optional PyMC
- Plotly + Streamlit
- `nba_api` for NBA.com endpoints
- Optional possession data from `shufinskiy/nba_data`
- Wikimedia Analytics API
- Manual, source-linked research ledgers

## 1. Create the environment

```bash
git init
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e '.[dev]'
cp .env.example .env
```

Or:

```bash
make setup
```

## 2. Run a smoke test first

NBA.com endpoints sometimes throttle or time out. Start without game logs:

```bash
goatlab ingest-core --skip-game-logs
```

If successful, run the complete ingestion:

```bash
goatlab ingest-core
```

Every response is cached in `data/raw/nba_api/`, so reruns do not repeat successful requests.

## 3. Add Basketball-Reference advanced metrics manually

Automated scraping is intentionally not the default. For both players, export the regular-season and playoff **Advanced** tables as CSV and normalize them to the schema in:

```text
data/manual/manual_advanced_template.csv
```

Then run:

```bash
goatlab import-advanced data/manual/manual_advanced.csv
```

Do the same for MVP voting shares:

```bash
goatlab import-mvp-votes data/manual/mvp_vote_shares.csv
```

Detailed instructions are in `docs/MANUAL_DATA_GUIDE.md`.

## 4. Add contextual research tables

Complete these templates before the final publication build:

- `data/manual/playoff_series.csv`
- `data/manual/impact_ledger.csv`
- `data/manual/film_annotations.csv`
- `data/manual/arguments.csv`
- `data/manual/sources.csv`
- `data/manual/manual_category_inputs.csv`

The first two players are the focus, but the playoff expectation model is trained on **all playoff series**, not only their series.

## 5. Fetch modern cultural-attention data

```bash
goatlab ingest-cultural
```

Wikimedia pageviews begin in July 2015. This is an overlapping-period attention measure, not a fair full-career comparison. Export Google Trends manually and keep the raw CSV in `data/raw/google_trends/`.

## 6. Build features and train models

```bash
goatlab build-features
goatlab train-models
pytest -q
```

Or:

```bash
make all
```

## 7. Launch the dashboard

```bash
streamlit run app/Home.py
```

Open the local URL printed by Streamlit.

## 8. Publish

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a Streamlit app from the repository.
3. Set the entrypoint to `app/Home.py`.
4. Commit only processed, publication-safe Parquet files needed by the dashboard.
5. Do not publish API secrets or massive raw play-by-play files.

### Docker or other hosts

A normal command is:

```bash
streamlit run app/Home.py --server.address 0.0.0.0 --server.port $PORT
```

## Reproducible release rule

Tag the public result with a frozen cutoff, for example:

```text
GOAT Lab v1.0 — data through the 2025-26 NBA season
```

Store the cutoff, source versions, file hashes, and model metadata. Never allow changing live data to silently alter an already published conclusion.

## Recommended final pages

1. Executive conclusion
2. Career curves
3. Peak, prime, and longevity
4. Scoring and playmaking
5. Defense
6. Playoffs and opponent quality
7. Team and teammate context
8. Skill versatility and portability
9. Cultural/commercial/social impact
10. Argument explorer
11. Weight simulator
12. Robustness and limitations
13. Sources and downloadable methodology

## Research honesty

Some modules cannot have equal data quality across eras:

- Official possession and lineup data begin late in Jordan’s career.
- Modern tracking data did not exist during Jordan’s career.
- Digital attention data begin long after Jordan’s peak.
- Historical injury, matchup, and defensive-assignment data require manual research and film coding.

GOAT Lab never converts those missing values to zero. It labels coverage, uses matched comparison windows where possible, and reruns the conclusion with modern-only metrics removed.
