from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


HIGHER_IS_BETTER = {
    "PTS": True,
    "AST": True,
    "REB": True,
    "STL": True,
    "BLK": True,
    "TS_PCT": True,
    "NET_RATING": True,
    "OFF_RATING": True,
    "DEF_RATING": False,
    "AST_PCT": True,
    "REB_PCT": True,
    "USG_PCT": True,
}


def add_true_shooting(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    denominator = 2 * (result.get("FGA", 0) + 0.44 * result.get("FTA", 0))
    result["TS_PCT_CALC"] = np.where(denominator > 0, result.get("PTS", 0) / denominator, np.nan)
    return result


def add_relative_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
    weight_column: str = "MIN",
) -> pd.DataFrame:
    """Add league-relative difference, z-score, percentile, and reliability.

    League statistics are weighted by minutes where possible. Percentiles are computed
    among player-seasons meeting the caller's qualification rule.
    """
    result = frame.copy()
    for metric in metric_columns:
        if metric not in result.columns:
            continue
        output_parts: list[pd.DataFrame] = []
        for _, group in result.groupby(group_columns, dropna=False):
            group = group.copy()
            values = pd.to_numeric(group[metric], errors="coerce")
            weights = pd.to_numeric(group.get(weight_column, 1), errors="coerce").fillna(0)
            valid = values.notna()
            if valid.sum() < 3:
                group[f"{metric}_REL"] = np.nan
                group[f"{metric}_Z"] = np.nan
                group[f"{metric}_PCTL"] = np.nan
                output_parts.append(group)
                continue
            if weights[valid].sum() > 0:
                mean = float(np.average(values[valid], weights=weights[valid]))
                variance = float(np.average((values[valid] - mean) ** 2, weights=weights[valid]))
                std = variance**0.5
            else:
                mean = float(values[valid].mean())
                std = float(values[valid].std(ddof=0))
            direction = 1 if HIGHER_IS_BETTER.get(metric, True) else -1
            group[f"{metric}_REL"] = direction * (values - mean)
            group[f"{metric}_Z"] = direction * ((values - mean) / std if std > 0 else np.nan)
            group[f"{metric}_PCTL"] = values.apply(
                lambda value: (
                    percentileofscore(values[valid], value, kind="rank")
                    if pd.notna(value)
                    else np.nan
                )
            )
            if direction == -1:
                group[f"{metric}_PCTL"] = 100 - group[f"{metric}_PCTL"]
            output_parts.append(group)
        result = pd.concat(output_parts, ignore_index=True)
    return result


def shrink_z_scores(
    frame: pd.DataFrame,
    z_columns: list[str],
    minutes_column: str = "MIN",
    prior_minutes: float = 500.0,
) -> pd.DataFrame:
    """Shrink small-minute samples toward league average."""
    result = frame.copy()
    minutes = pd.to_numeric(result.get(minutes_column, 0), errors="coerce").fillna(0).clip(lower=0)
    reliability = minutes / (minutes + prior_minutes)
    result["RELIABILITY"] = reliability
    for column in z_columns:
        if column in result.columns:
            result[f"{column}_SHRUNK"] = result[column] * reliability
    return result
