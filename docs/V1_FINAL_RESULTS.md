# GOAT Lab v1 Final Results

## Headline

Under the preregistered v1 production model, **LeBron James ranks first by 0.115091 points**.

Across 250,000 hierarchy-aware weight simulations, **LeBron James wins 60.1484%** of sampled value systems.

> This is a conditional model result, not an objective probability that either player is the GOAT.

## Frozen central result

| Player | GOAT score | Rank |
|---|---:|---:|
| LeBron James | 89.258985 | 1 |
| Michael Jordan | 89.143895 | 2 |

## Preregistered simulation

| Player | Win rate | Mean score | P05 | P95 |
|---|---:|---:|---:|---:|
| LeBron James | 60.1484% | 89.257365 | 88.217299 | 90.289118 |
| Michael Jordan | 39.8516% | 89.142931 | 88.165156 | 90.048420 |

The win rate is the share of frozen-cap, within-group weight systems won. Group mass remains exactly 50% Performance Arc, 40% Basketball Value, and 10% Broader Legacy.

## Scale sensitivity

| Scaling scenario | LeBron | Jordan | L-J margin | Winner |
|---|---:|---:|---:|---|
| bounded_logit_tail *(production)* | 89.258985 | 89.143895 | +0.115091 | LeBron James |
| historical_percentile | 90.730646 | 93.313241 | -2.582595 | Michael Jordan |
| normal_score_tail | 87.177390 | 84.951779 | +2.225611 | LeBron James |
| robust_mad_reference | 80.078266 | 80.364457 | -0.286191 | Michael Jordan |

**Robustness conclusion:** the winner is not stable across the four approved scaling scenarios. The production result uses `bounded_logit_tail` because that method was frozen before the final simulation.

## Largest simulation drivers

| Category | Frozen weight | Margin correlation | Interpretation |
|---|---:|---:|---|
| defense | 0.120 | -0.975068 | More weight generally favors Jordan. |
| offense | 0.180 | +0.727762 | More weight generally favors LeBron. |
| winning_context | 0.100 | +0.194225 | More weight generally favors LeBron. |
| prime | 0.100 | -0.096294 | More weight generally favors Jordan. |
| longevity | 0.075 | +0.069637 | More weight generally favors LeBron. |

Defense is the strongest swing factor, while offense is the largest counterweight favoring LeBron.

## Evidence treatment and limitations

- The central winner changes across approved category-scaling scenarios.
- The cultural-impact ordering changes across reasonable weighting choices.
- Expert-film evidence was excluded from the primary model because no source met the frozen eligibility standard.
- Game-level playoff, impact-metric, and supporting-cast audits remain diagnostic and add zero central weight.
- The simulation varies within-group category weights around fixed group caps; it does not sample every possible GOAT philosophy.

## Reproducibility

- Source commit: `57e504601898afe4e8ead2fa1e51d25990b47de2`
- Source branch: `feature/model-integrity`
- Production scale: `bounded_logit_tail`
- Simulations: `250000`
- Random seed: `23`
- Within-group concentration: `100.0`
- Release gate: `32/32` checks passed
- Artifact hashes: `release/v1_artifact_hashes.sha256`
- Machine-readable manifest: `release/v1_release_manifest.json`
