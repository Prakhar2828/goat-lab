from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {
    "peak",
    "prime",
    "longevity",
    "regular_season",
    "playoffs",
    "winning_context",
    "offense",
    "defense",
    "cultural_impact",
}


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    unknown = set(weights) - REQUIRED_SCORE_COLUMNS
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    clean = {category: max(float(weights.get(category, 0.0)), 0.0) for category in REQUIRED_SCORE_COLUMNS}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("At least one category weight must be positive.")
    return {category: value / total for category, value in clean.items()}


def score_players(category_scores: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    normalized = validate_weights(weights)
    result = category_scores.copy()
    available_categories = [category for category in normalized if category in result.columns]
    if not available_categories:
        raise ValueError("No score columns match the selected categories.")

    def row_score(row: pd.Series) -> float:
        available = {
            category: normalized[category]
            for category in available_categories
            if pd.notna(row[category])
        }
        if not available:
            return float("nan")
        weight_sum = sum(available.values())
        return float(sum(row[category] * weight for category, weight in available.items()) / weight_sum)

    result["GOAT_SCORE"] = result.apply(row_score, axis=1)
    result["RANK"] = result["GOAT_SCORE"].rank(ascending=False, method="min").astype("Int64")
    return result.sort_values("GOAT_SCORE", ascending=False)


def normalize_category_scores(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        minimum, maximum = values.min(), values.max()
        if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
            result[column] = 50.0
        else:
            result[column] = 100 * (values - minimum) / (maximum - minimum)
    return result
