from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FamilySpec:
    name: str
    columns: tuple[str, ...]
    weights: tuple[float, ...] | None = None


DEFAULT_FAMILIES = (
    FamilySpec("scoring", ("PTS_PER75_Z_SHRUNK", "TS_PCT_Z_SHRUNK", "USG_PCT_Z_SHRUNK")),
    FamilySpec("playmaking", ("AST_PER75_Z_SHRUNK", "AST_PCT_Z_SHRUNK")),
    FamilySpec("rebounding", ("REB_PER75_Z_SHRUNK", "REB_PCT_Z_SHRUNK")),
    FamilySpec("defense_box", ("STL_PER75_Z_SHRUNK", "BLK_PER75_Z_SHRUNK", "DEF_RATING_Z_SHRUNK")),
    FamilySpec("team_impact", ("NET_RATING_Z_SHRUNK", "PIE_Z_SHRUNK")),
)


def coverage_aware_average(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True)


def build_family_scores(
    frame: pd.DataFrame,
    families: tuple[FamilySpec, ...] = DEFAULT_FAMILIES,
) -> pd.DataFrame:
    result = frame.copy()
    for family in families:
        available = [column for column in family.columns if column in result.columns]
        if not available:
            result[f"FAMILY_{family.name.upper()}"] = np.nan
            result[f"COVERAGE_{family.name.upper()}"] = 0.0
            continue
        values = result[available].apply(pd.to_numeric, errors="coerce")
        result[f"FAMILY_{family.name.upper()}"] = values.mean(axis=1, skipna=True)
        result[f"COVERAGE_{family.name.upper()}"] = values.notna().mean(axis=1)
    return result


def fit_latent_season_score(
    league_frame: pd.DataFrame,
    feature_columns: list[str],
    minutes_column: str = "MIN",
) -> tuple[pd.Series, dict[str, float]]:
    """Fit a one-component PCA latent score after median imputation.

    PCA is used only as a robustness model to reduce double-counting of correlated
    metrics. The primary public score remains a transparent family-weighted model.
    """
    available = [column for column in feature_columns if column in league_frame.columns]
    if len(available) < 2:
        return pd.Series(np.nan, index=league_frame.index), {}

    matrix = league_frame[available].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median())
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    weights = pd.to_numeric(league_frame.get(minutes_column, 1), errors="coerce").fillna(1)
    pca = PCA(n_components=1, random_state=23)
    scores = pca.fit_transform(standardized).ravel()

    loadings = dict(zip(available, pca.components_[0], strict=False))
    if sum(loadings.values()) < 0:
        scores *= -1
        loadings = {column: -value for column, value in loadings.items()}

    # Weighting does not affect sklearn PCA fit directly; report a minute-based
    # reliability shrinkage to avoid tiny samples dominating.
    reliability = weights / (weights + 500.0)
    return pd.Series(scores * reliability, index=league_frame.index), loadings


def add_transparent_season_value(frame: pd.DataFrame) -> pd.DataFrame:
    result = build_family_scores(frame)
    family_columns = [column for column in result.columns if column.startswith("FAMILY_")]
    result["SEASON_VALUE_Z"] = result[family_columns].mean(axis=1, skipna=True)
    result["SEASON_VALUE_0_100"] = 50 + 15 * result["SEASON_VALUE_Z"]
    result["SEASON_VALUE_0_100"] = result["SEASON_VALUE_0_100"].clip(0, 100)
    return result
