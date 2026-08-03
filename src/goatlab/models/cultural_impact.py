from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from goatlab.settings import settings
from goatlab.utils import (
    load_yaml,
    read_optional_parquet,
    write_parquet,
)


DEFAULT_ATTENTION_COMPONENTS = {
    "total_view_share": 0.50,
    "median_daily_view_share": 0.30,
    "median_annual_view_share": 0.20,
}

DEFAULT_DIMENSIONS = {
    "commercial_global_reach": 0.30,
    "basketball_culture_influence": 0.30,
    "media_entertainment_reach": 0.15,
    "philanthropy_social_institutions": 0.25,
}

DEFAULT_CONFIDENCE_VALUES = {
    "High": 1.00,
    "Medium": 0.75,
    "Low": 0.50,
}


def _normalize_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    numeric = {
        str(name): float(value)
        for name, value in weights.items()
    }

    if not numeric:
        raise ValueError(
            "At least one weight is required."
        )

    if any(
        not np.isfinite(value) or value < 0
        for value in numeric.values()
    ):
        raise ValueError(
            "Weights must be finite and non-negative."
        )

    total = float(sum(numeric.values()))

    if total <= 0:
        raise ValueError(
            "Weights must have a positive total."
        )

    return {
        name: value / total
        for name, value in numeric.items()
    }


def _safe_share(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    total = float(
        numeric.fillna(0).sum()
    )

    if total <= 0:
        return pd.Series(
            np.nan,
            index=values.index,
            dtype="float64",
        )

    return numeric / total


def calculate_attention_scores(
    pageviews: pd.DataFrame,
    component_weights: Mapping[
        str,
        float,
    ] | None = None,
) -> pd.DataFrame:
    """Calculate a robust common-window attention score.

    The score combines total view share, median daily view share,
    and median annual view share. Using all three prevents one major
    attention spike from controlling the entire component.
    """

    required = {
        "PLAYER_NAME",
        "date",
        "views",
    }

    missing = required.difference(
        pageviews.columns
    )

    if missing:
        raise ValueError(
            "Pageview data is missing columns: "
            f"{sorted(missing)}"
        )

    frame = pageviews.copy()

    frame["PLAYER_NAME"] = (
        frame["PLAYER_NAME"]
        .astype(str)
        .str.strip()
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["views"] = pd.to_numeric(
        frame["views"],
        errors="coerce",
    )

    if frame[
        [
            "PLAYER_NAME",
            "date",
            "views",
        ]
    ].isna().any().any():
        raise ValueError(
            "Pageview data contains missing or invalid values."
        )

    if (frame["views"] < 0).any():
        raise ValueError(
            "Pageview counts cannot be negative."
        )

    duplicates = frame.duplicated(
        [
            "PLAYER_NAME",
            "date",
        ]
    )

    if duplicates.any():
        raise ValueError(
            "Pageview data contains duplicate player-date rows."
        )

    coverage = (
        frame.groupby(
            "PLAYER_NAME",
            as_index=False,
        )
        .agg(
            START_DATE=("date", "min"),
            END_DATE=("date", "max"),
        )
    )

    if len(coverage) < 2:
        raise ValueError(
            "At least two players are required "
            "for comparative attention scoring."
        )

    common_start = coverage[
        "START_DATE"
    ].max()

    common_end = coverage[
        "END_DATE"
    ].min()

    if (
        pd.isna(common_start)
        or pd.isna(common_end)
        or common_start > common_end
    ):
        raise ValueError(
            "The players do not share a valid comparison window."
        )

    common = frame[
        frame["date"].between(
            common_start,
            common_end,
        )
    ].copy()

    total_views = (
        common.groupby(
            "PLAYER_NAME"
        )["views"]
        .sum()
        .rename("TOTAL_VIEWS")
    )

    median_daily = (
        common.groupby(
            "PLAYER_NAME"
        )["views"]
        .median()
        .rename("MEDIAN_DAILY_VIEWS")
    )

    annual = (
        common.assign(
            YEAR=common["date"].dt.year
        )
        .groupby(
            [
                "YEAR",
                "PLAYER_NAME",
            ],
            as_index=False,
        )["views"]
        .sum()
    )

    annual["ANNUAL_VIEW_SHARE"] = (
        annual["views"]
        / annual.groupby("YEAR")[
            "views"
        ].transform("sum")
    )

    median_annual_share = (
        annual.groupby(
            "PLAYER_NAME"
        )["ANNUAL_VIEW_SHARE"]
        .median()
        .rename(
            "MEDIAN_ANNUAL_VIEW_SHARE"
        )
    )

    result = pd.concat(
        [
            total_views,
            median_daily,
            median_annual_share,
        ],
        axis=1,
    ).reset_index()

    result["TOTAL_VIEW_SHARE"] = (
        _safe_share(
            result["TOTAL_VIEWS"]
        )
    )

    result[
        "MEDIAN_DAILY_VIEW_SHARE"
    ] = _safe_share(
        result["MEDIAN_DAILY_VIEWS"]
    )

    result[
        "MEDIAN_ANNUAL_VIEW_SHARE"
    ] = _safe_share(
        result[
            "MEDIAN_ANNUAL_VIEW_SHARE"
        ]
    )

    weights = _normalize_weights(
        component_weights
        or DEFAULT_ATTENTION_COMPONENTS
    )

    required_weight_names = {
        "total_view_share",
        "median_daily_view_share",
        "median_annual_view_share",
    }

    if set(weights) != required_weight_names:
        raise ValueError(
            "Attention weights must contain exactly: "
            f"{sorted(required_weight_names)}"
        )

    result["ATTENTION_SCORE"] = 100 * (
        weights["total_view_share"]
        * result["TOTAL_VIEW_SHARE"]
        + weights[
            "median_daily_view_share"
        ]
        * result[
            "MEDIAN_DAILY_VIEW_SHARE"
        ]
        + weights[
            "median_annual_view_share"
        ]
        * result[
            "MEDIAN_ANNUAL_VIEW_SHARE"
        ]
    )

    result["COMMON_START_DATE"] = (
        common_start
    )

    result["COMMON_END_DATE"] = (
        common_end
    )

    return result.sort_values(
        "PLAYER_NAME"
    ).reset_index(drop=True)


def calculate_rubric_scores(
    rubric: pd.DataFrame,
    dimension_weights: Mapping[
        str,
        float,
    ] | None = None,
    confidence_values: Mapping[
        str,
        float,
    ] | None = None,
) -> pd.DataFrame:
    """Calculate manually sourced rubric scores.

    A final rubric score is produced only when every required
    dimension has a valid 0-100 score.
    """

    required = {
        "PLAYER_NAME",
        "DIMENSION",
        "SCORE_0_100",
        "CONFIDENCE",
        "SOURCE_IDS",
        "RATIONALE",
    }

    missing = required.difference(
        rubric.columns
    )

    if missing:
        raise ValueError(
            "Cultural rubric is missing columns: "
            f"{sorted(missing)}"
        )

    dimensions = _normalize_weights(
        dimension_weights
        or DEFAULT_DIMENSIONS
    )

    confidence_map = {
        str(name).strip().title(): float(value)
        for name, value in (
            confidence_values
            or DEFAULT_CONFIDENCE_VALUES
        ).items()
    }

    frame = rubric.copy()

    frame["PLAYER_NAME"] = (
        frame["PLAYER_NAME"]
        .astype(str)
        .str.strip()
    )

    frame["DIMENSION"] = (
        frame["DIMENSION"]
        .astype(str)
        .str.strip()
    )

    frame["SCORE_0_100"] = (
        pd.to_numeric(
            frame["SCORE_0_100"],
            errors="coerce",
        )
    )

    frame["CONFIDENCE"] = (
        frame["CONFIDENCE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    duplicate_mask = frame.duplicated(
        [
            "PLAYER_NAME",
            "DIMENSION",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicated = frame.loc[
            duplicate_mask,
            [
                "PLAYER_NAME",
                "DIMENSION",
            ],
        ]

        raise ValueError(
            "Duplicate player-dimension rubric rows:\n"
            + duplicated.to_string(
                index=False
            )
        )

    invalid_scores = (
        frame["SCORE_0_100"].notna()
        & ~frame["SCORE_0_100"].between(
            0,
            100,
        )
    )

    if invalid_scores.any():
        raise ValueError(
            "Cultural rubric scores must be "
            "between 0 and 100."
        )

    nonblank_confidence = frame[
        "CONFIDENCE"
    ].ne("")

    unknown_confidence = (
        nonblank_confidence
        & ~frame["CONFIDENCE"].isin(
            confidence_map
        )
    )

    if unknown_confidence.any():
        values = sorted(
            frame.loc[
                unknown_confidence,
                "CONFIDENCE",
            ].unique()
        )

        raise ValueError(
            "Unknown confidence values: "
            f"{values}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for player_name, group in frame.groupby(
        "PLAYER_NAME"
    ):
        by_dimension = group.set_index(
            "DIMENSION"
        )

        weighted_score = 0.0
        valid_weight = 0.0
        confidence_total = 0.0

        missing_dimensions: list[str] = []

        for dimension, weight in dimensions.items():
            if dimension not in by_dimension.index:
                missing_dimensions.append(
                    dimension
                )
                continue

            row = by_dimension.loc[
                dimension
            ]

            score = row[
                "SCORE_0_100"
            ]

            if pd.isna(score):
                missing_dimensions.append(
                    dimension
                )
                continue

            weighted_score += (
                float(score)
                * weight
            )

            valid_weight += weight

            confidence_name = str(
                row["CONFIDENCE"]
            ).strip().title()

            confidence_total += (
                confidence_map.get(
                    confidence_name,
                    0.0,
                )
                * weight
            )

        coverage = float(
            valid_weight
        )

        complete = (
            not missing_dimensions
            and np.isclose(
                coverage,
                1.0,
            )
        )

        rubric_score = (
            weighted_score
            if complete
            else float("nan")
        )

        rubric_confidence = (
            confidence_total
            / valid_weight
            if valid_weight > 0
            else float("nan")
        )

        rows.append(
            {
                "PLAYER_NAME": player_name,
                "RUBRIC_SCORE": rubric_score,
                "RUBRIC_COVERAGE": coverage,
                "RUBRIC_CONFIDENCE": (
                    rubric_confidence
                ),
                "RUBRIC_COMPLETE": complete,
                "MISSING_DIMENSIONS": (
                    ",".join(
                        missing_dimensions
                    )
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "PLAYER_NAME"
    ).reset_index(drop=True)


def combine_cultural_scores(
    attention: pd.DataFrame,
    rubric: pd.DataFrame,
    attention_weight: float = 0.20,
    rubric_weight: float = 0.80,
) -> pd.DataFrame:
    weights = _normalize_weights(
        {
            "attention": attention_weight,
            "rubric": rubric_weight,
        }
    )

    result = attention.merge(
        rubric,
        on="PLAYER_NAME",
        how="outer",
        validate="one_to_one",
    )

    complete = (
        result["ATTENTION_SCORE"].notna()
        & result["RUBRIC_COMPLETE"]
        .fillna(False)
        .astype(bool)
        & result["RUBRIC_SCORE"].notna()
    )

    result[
        "cultural_impact_raw"
    ] = np.where(
        complete,
        (
            weights["attention"]
            * result["ATTENTION_SCORE"]
            + weights["rubric"]
            * result["RUBRIC_SCORE"]
        ),
        np.nan,
    )

    result[
        "CULTURAL_INPUT_COVERAGE"
    ] = (
        weights["attention"]
        * result["ATTENTION_SCORE"]
        .notna()
        .astype(float)
        + weights["rubric"]
        * result["RUBRIC_COVERAGE"]
        .fillna(0)
    )

    result[
        "CULTURAL_SCORE_COMPLETE"
    ] = complete

    return result.sort_values(
        "PLAYER_NAME"
    ).reset_index(drop=True)


def build_cultural_impact_scores(
    config_path: str | Path = (
        "configs/cultural_impact.yaml"
    ),
) -> pd.DataFrame:
    pageviews_path = (
        settings.interim_dir
        / "wikimedia_pageviews.parquet"
    )

    rubric_path = (
        settings.manual_dir
        / "cultural_rubric.csv"
    )

    pageviews = read_optional_parquet(
        pageviews_path
    )

    if pageviews.empty:
        raise FileNotFoundError(
            "Wikimedia pageviews are missing. "
            "Run `goatlab ingest-cultural` first."
        )

    if not rubric_path.exists():
        raise FileNotFoundError(
            f"Missing cultural rubric: {rubric_path}"
        )

    rubric = pd.read_csv(
        rubric_path
    )

    config = load_yaml(
        config_path
    )

    attention = calculate_attention_scores(
        pageviews,
        component_weights=config.get(
            "attention_components",
            DEFAULT_ATTENTION_COMPONENTS,
        ),
    )

    rubric_scores = calculate_rubric_scores(
        rubric,
        dimension_weights=config.get(
            "rubric_dimensions",
            DEFAULT_DIMENSIONS,
        ),
        confidence_values=config.get(
            "confidence_values",
            DEFAULT_CONFIDENCE_VALUES,
        ),
    )

    combined = combine_cultural_scores(
        attention,
        rubric_scores,
        attention_weight=float(
            config.get(
                "attention_weight",
                0.20,
            )
        ),
        rubric_weight=float(
            config.get(
                "rubric_weight",
                0.80,
            )
        ),
    )

    write_parquet(
        combined,
        settings.processed_dir
        / "cultural_impact_scores.parquet",
    )

    return combined
