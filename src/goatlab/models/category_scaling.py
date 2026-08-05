from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore

REFERENCE_CATEGORIES: Mapping[str, str] = {
    "peak": "peak_raw",
    "prime": "prime_raw",
    "longevity": "longevity_raw",
    "regular_season": "regular_season_raw",
    "playoffs": "playoffs_raw",
    "offense": "offense_raw",
    "defense": "defense_raw",
}

NATIVE_SCALE_CATEGORIES = (
    "winning_context",
    "cultural_impact",
)

SCALING_SCENARIOS = (
    "historical_percentile",
    "normal_score_tail",
    "bounded_logit_tail",
    "robust_mad_reference",
)

_MIN_PROBABILITY = 0.001
_MAX_PROBABILITY = 0.999
_ROBUST_Z_SPAN = 6.0
_LOGIT_CURVATURE = 4.0


def reference_percentile(
    value: float,
    reference: pd.Series,
) -> float:
    """Return an empirical rank percentile on a 0-100 scale."""
    clean = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    if pd.isna(value) or clean.empty:
        return float("nan")

    return float(
        percentileofscore(
            clean,
            float(value),
            kind="rank",
        )
    )


def percentile_to_normal_score(
    values: pd.Series,
) -> pd.Series:
    """Expand percentile tails with a bounded normal-score transform."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    probability = (
        numeric / 100.0
    ).clip(
        _MIN_PROBABILITY,
        _MAX_PROBABILITY,
    )

    normal_value = pd.Series(
        norm.ppf(probability),
        index=values.index,
        dtype="float64",
    )

    anchor = float(
        norm.ppf(_MAX_PROBABILITY)
    )

    transformed = (
        50.0
        + 50.0
        * normal_value
        / anchor
    ).clip(
        0.0,
        100.0,
    )

    return transformed.where(
        numeric.notna()
    )


def percentile_to_bounded_logit_score(
    values: pd.Series,
) -> pd.Series:
    """Expand elite-percentile separation without an unbounded score."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    probability = (
        numeric / 100.0
    ).clip(
        _MIN_PROBABILITY,
        _MAX_PROBABILITY,
    )

    log_odds = np.log(
        probability
        / (1.0 - probability)
    )

    anchor_log_odds = float(
        np.log(
            _MAX_PROBABILITY
            / (1.0 - _MAX_PROBABILITY)
        )
    )

    denominator = float(
        np.tanh(
            anchor_log_odds
            / _LOGIT_CURVATURE
        )
    )

    transformed = (
        50.0
        + 50.0
        * np.tanh(
            log_odds
            / _LOGIT_CURVATURE
        )
        / denominator
    ).clip(
        0.0,
        100.0,
    )

    return pd.Series(
        transformed,
        index=values.index,
        dtype="float64",
    ).where(
        numeric.notna()
    )


def robust_reference_parameters(
    reference: pd.Series,
) -> tuple[float, float, str]:
    """Return a robust center and nonzero scale with documented fallback."""
    clean = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna().astype(float)

    if clean.empty:
        return (
            float("nan"),
            float("nan"),
            "missing_reference",
        )

    center = float(
        clean.median()
    )

    mad = float(
        np.median(
            np.abs(
                clean.to_numpy()
                - center
            )
        )
    )

    mad_scale = (
        1.4826 * mad
    )

    if (
        np.isfinite(mad_scale)
        and mad_scale > 1e-12
    ):
        return (
            center,
            float(mad_scale),
            "median_mad",
        )

    q25 = float(
        clean.quantile(0.25)
    )
    q75 = float(
        clean.quantile(0.75)
    )
    iqr_scale = (
        (q75 - q25)
        / 1.349
    )

    if (
        np.isfinite(iqr_scale)
        and iqr_scale > 1e-12
    ):
        return (
            center,
            float(iqr_scale),
            "median_iqr_fallback",
        )

    standard_deviation = float(
        clean.std(
            ddof=0
        )
    )

    if (
        np.isfinite(
            standard_deviation
        )
        and standard_deviation
        > 1e-12
    ):
        return (
            center,
            standard_deviation,
            "median_std_fallback",
        )

    return (
        center,
        1.0,
        "constant_reference_fallback",
    )


def robust_mad_reference_score(
    value: float,
    reference: pd.Series,
) -> tuple[float, float, float, str]:
    """Map a raw value through a bounded robust z-score transformation."""
    if pd.isna(value):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            "missing_value",
        )

    center, scale, method = (
        robust_reference_parameters(
            reference
        )
    )

    if (
        not np.isfinite(center)
        or not np.isfinite(scale)
    ):
        return (
            float("nan"),
            center,
            scale,
            method,
        )

    robust_z = (
        float(value) - center
    ) / scale

    score = (
        50.0
        + 50.0
        * np.tanh(
            robust_z
            / _ROBUST_Z_SPAN
        )
    )

    return (
        float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        ),
        center,
        scale,
        method,
    )


def transform_percentile_scores(
    values: pd.Series,
    scenario: str,
) -> pd.Series:
    """Apply a declared percentile-based scale transformation."""
    if scenario == "historical_percentile":
        return pd.to_numeric(
            values,
            errors="coerce",
        )

    if scenario == "normal_score_tail":
        return percentile_to_normal_score(
            values
        )

    if scenario == "bounded_logit_tail":
        return (
            percentile_to_bounded_logit_score(
                values
            )
        )

    raise ValueError(
        "Scenario requires raw reference values "
        f"or is unknown: {scenario!r}."
    )


def build_scaling_comparison(
    category_scores: pd.DataFrame,
    historical_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Build target-player scores under every approved diagnostic scale."""
    required_scores = {
        "PLAYER_NAME",
        *REFERENCE_CATEGORIES,
        *NATIVE_SCALE_CATEGORIES,
    }
    missing_scores = (
        required_scores.difference(
            category_scores.columns
        )
    )

    if missing_scores:
        raise ValueError(
            "Category scores are missing columns: "
            f"{sorted(missing_scores)}"
        )

    required_reference = {
        "PLAYER_NAME",
        *REFERENCE_CATEGORIES.values(),
    }
    missing_reference = (
        required_reference.difference(
            historical_reference.columns
        )
    )

    if missing_reference:
        raise ValueError(
            "Historical reference is missing columns: "
            f"{sorted(missing_reference)}"
        )

    raw_targets = (
        historical_reference[
            historical_reference[
                "PLAYER_NAME"
            ].isin(
                category_scores[
                    "PLAYER_NAME"
                ]
            )
        ]
        .drop_duplicates(
            subset=[
                "PLAYER_NAME",
            ],
            keep="last",
        )
        .set_index(
            "PLAYER_NAME"
        )
    )

    score_index = (
        category_scores.set_index(
            "PLAYER_NAME"
        )
    )

    missing_players = set(
        score_index.index
    ).difference(
        raw_targets.index
    )

    if missing_players:
        raise ValueError(
            "Target players are missing raw "
            "reference rows: "
            f"{sorted(missing_players)}"
        )

    rows: list[
        dict[
            str,
            float | int | str,
        ]
    ] = []

    for player_name in (
        score_index.index.astype(str)
    ):
        for (
            category,
            raw_column,
        ) in REFERENCE_CATEGORIES.items():
            percentile_score = float(
                score_index.loc[
                    player_name,
                    category,
                ]
            )
            raw_value = float(
                raw_targets.loc[
                    player_name,
                    raw_column,
                ]
            )
            reference_values = (
                historical_reference[
                    raw_column
                ]
            )

            robust_score, center, scale, method = (
                robust_mad_reference_score(
                    raw_value,
                    reference_values,
                )
            )

            scenario_scores = {
                "historical_percentile": (
                    percentile_score
                ),
                "normal_score_tail": float(
                    percentile_to_normal_score(
                        pd.Series(
                            [
                                percentile_score,
                            ]
                        )
                    ).iloc[0]
                ),
                "bounded_logit_tail": float(
                    percentile_to_bounded_logit_score(
                        pd.Series(
                            [
                                percentile_score,
                            ]
                        )
                    ).iloc[0]
                ),
                "robust_mad_reference": (
                    robust_score
                ),
            }

            for (
                scenario,
                score,
            ) in scenario_scores.items():
                rows.append(
                    {
                        "SCENARIO": scenario,
                        "PLAYER_NAME": (
                            player_name
                        ),
                        "CATEGORY": category,
                        "RAW_VALUE": raw_value,
                        "PERCENTILE_SCORE": (
                            percentile_score
                        ),
                        "SCORE": score,
                        "REFERENCE_CENTER": (
                            center
                            if scenario
                            == "robust_mad_reference"
                            else np.nan
                        ),
                        "REFERENCE_SCALE": (
                            scale
                            if scenario
                            == "robust_mad_reference"
                            else np.nan
                        ),
                        "REFERENCE_METHOD": (
                            method
                            if scenario
                            == "robust_mad_reference"
                            else (
                                "empirical_cdf"
                                if scenario
                                == "historical_percentile"
                                else (
                                    "percentile_transform"
                                )
                            )
                        ),
                        "REFERENCE_SIZE": int(
                            pd.to_numeric(
                                reference_values,
                                errors="coerce",
                            )
                            .dropna()
                            .shape[0]
                        ),
                    }
                )

        for category in (
            NATIVE_SCALE_CATEGORIES
        ):
            native_score = float(
                score_index.loc[
                    player_name,
                    category,
                ]
            )

            for scenario in (
                SCALING_SCENARIOS
            ):
                rows.append(
                    {
                        "SCENARIO": scenario,
                        "PLAYER_NAME": (
                            player_name
                        ),
                        "CATEGORY": category,
                        "RAW_VALUE": native_score,
                        "PERCENTILE_SCORE": np.nan,
                        "SCORE": native_score,
                        "REFERENCE_CENTER": np.nan,
                        "REFERENCE_SCALE": np.nan,
                        "REFERENCE_METHOD": (
                            "native_evidence_scale"
                        ),
                        "REFERENCE_SIZE": 0,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    numeric_scores = pd.to_numeric(
        result["SCORE"],
        errors="coerce",
    )

    if (
        numeric_scores.isna().any()
        or not np.isfinite(
            numeric_scores
        ).all()
    ):
        raise ValueError(
            "Scaling comparison produced "
            "non-finite scores."
        )

    if not numeric_scores.between(
        0.0,
        100.0,
        inclusive="both",
    ).all():
        raise ValueError(
            "Scaling comparison produced "
            "scores outside [0, 100]."
        )

    return (
        result.sort_values(
            [
                "SCENARIO",
                "PLAYER_NAME",
                "CATEGORY",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_saturation_audit(
    comparison: pd.DataFrame,
    high_threshold: float = 98.0,
    minimum_separation: float = 1.0,
) -> pd.DataFrame:
    """Summarize compression and elite-tail saturation by scale/category."""
    required = {
        "SCENARIO",
        "PLAYER_NAME",
        "CATEGORY",
        "SCORE",
    }
    missing = required.difference(
        comparison.columns
    )

    if missing:
        raise ValueError(
            "Scaling comparison is missing columns: "
            f"{sorted(missing)}"
        )

    rows: list[
        dict[
            str,
            bool | float | int | str,
        ]
    ] = []

    for (
        scenario,
        category,
    ), group in comparison.groupby(
        [
            "SCENARIO",
            "CATEGORY",
        ],
        sort=True,
    ):
        scores = pd.to_numeric(
            group["SCORE"],
            errors="coerce",
        ).dropna()

        if scores.empty:
            continue

        score_range = float(
            scores.max()
            - scores.min()
        )
        high_count = int(
            scores.ge(
                high_threshold
            ).sum()
        )
        saturation_rate = float(
            high_count
            / len(scores)
        )
        standard_deviation = float(
            scores.std(
                ddof=0
            )
        )

        rows.append(
            {
                "SCENARIO": str(
                    scenario
                ),
                "CATEGORY": str(
                    category
                ),
                "PLAYER_COUNT": len(scores),
                "MIN_SCORE": float(
                    scores.min()
                ),
                "MAX_SCORE": float(
                    scores.max()
                ),
                "SCORE_RANGE": (
                    score_range
                ),
                "SCORE_STD": (
                    standard_deviation
                ),
                "HIGH_SCORE_COUNT": (
                    high_count
                ),
                "HIGH_SCORE_RATE": (
                    saturation_rate
                ),
                "COMPRESSED_PAIR": bool(
                    len(scores) >= 2
                    and score_range
                    < minimum_separation
                ),
                "ELITE_TAIL_SATURATED": bool(
                    len(scores) >= 2
                    and saturation_rate
                    >= 0.5
                ),
                "SATURATION_FLAG": bool(
                    len(scores) >= 2
                    and (
                        score_range
                        < minimum_separation
                        or saturation_rate
                        >= 0.5
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "SCENARIO",
            "CATEGORY",
        ]
    ).reset_index(
        drop=True
    )
