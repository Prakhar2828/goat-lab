# Model card

## Overall system

GOAT Lab is an explanatory decision-support system. It is not designed to predict future NBA outcomes or replace expert judgment.

## Model 1 — Transparent season value

**Purpose:** summarize season-level evidence while preserving interpretable metric families.

**Inputs:** era-standardized and reliability-shrunk scoring, playmaking, rebounding, defense, and team-impact features.

**Output:** season value on a display scale plus family subscores.

**Main risk:** family construction and weights remain value judgments.

**Required robustness:** equal family weights, alternative family weights, correlated-metric removal, and PCA comparison.

## Model 2 — PCA latent season factor

**Purpose:** detect whether the transparent result is driven by double counting correlated statistics.

**Inputs:** standardized league-wide performance features.

**Output:** first principal-component score and loadings.

**Limitations:** PCA maximizes variance, not basketball truth. Loadings can change by era and feature set.

## Model 3 — Playoff-series expectation

**Purpose:** estimate the probability of winning a playoff series given pre-series team context.

**Target:** team won series.

**Baseline:** regularized logistic regression.

**Training population:** all team-sides of NBA playoff series in the selected historical window.

**Validation:** chronological holdout by season; report ROC-AUC, log loss, Brier score, and calibration.

**Leakage exclusions:** final series result, games won, in-series injuries that occurred after series start, post-series statistics, and final playoff-round outcome.

**Interpretation:** actual minus expected outcome describes team-level overperformance. It does not assign all credit to one player.

## Model 4 — RAPM / lineup model

**Purpose:** estimate player effects from possession-level lineup data.

**Method:** ridge regression of possession point differential on offensive and defensive player indicators, with season and home-court controls.

**Coverage limitation:** public possession data does not cover Jordan’s full career. This model must remain supplementary or matched-window-only.

## Model 5 — Cultural latent index

**Purpose:** summarize multiple forms of public and broader impact without adding incomparable raw units.

**Inputs:** standardized attention, commercial, social-impact, and player-influence indicators.

**Risks:** measurement availability strongly favors modern digital eras; commercial estimates can be proprietary; philanthropic announcements may not equal outcomes.

**Safeguards:** separate overlapping-period and legacy indexes; show raw evidence ledger; no missing-as-zero scoring.

## Intended users

Basketball fans, sports-analytics readers, recruiters reviewing the project, and researchers interested in transparent multi-criteria analysis.

## Not intended for

Betting, player employment decisions, automated Hall of Fame voting, or claims of causal social impact.

## Publication requirements

- Data cutoff shown on every release
- Source registry included
- Model metrics included
- Missingness and coverage report included
- Reproducible seed and configuration included
- No result published without sensitivity analysis
