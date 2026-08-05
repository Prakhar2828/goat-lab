from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore

from goatlab.models.peak_prime import (
    summarize_peak_prime_longevity,
)

CATEGORIES = (
    "peak",
    "prime",
    "longevity",
    "regular_season",
    "playoffs",
    "winning_context",
    "offense",
    "defense",
    "cultural_impact",
)

REFERENCE_PERCENTILE_CATEGORIES = (
    "peak",
    "prime",
    "longevity",
    "regular_season",
    "playoffs",
    "offense",
    "defense",
)

AVAILABILITY_SENSITIVE_CATEGORIES = {
    "peak": "peak_raw",
    "prime": "prime_raw",
    "longevity": "longevity_raw",
    "regular_season": "regular_season_raw",
    "playoffs": "playoffs_raw",
}

ROUND_WEIGHT_SCENARIOS: dict[
    str,
    dict[int, float],
] = {
    "equal_series": {
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00,
    },
    "mild_late_round": {
        1: 1.00,
        2: 1.15,
        3: 1.35,
        4: 1.60,
    },
    "linear_round": {
        1: 1.00,
        2: 2.00,
        3: 3.00,
        4: 4.00,
    },
}

SCALE_SCENARIOS = (
    "historical_percentile",
    "normal_score_tail",
)


def _regular_season_mask(
    values: pd.Series,
) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("regular season")
    )


def _weighted_mean(
    frame: pd.DataFrame,
    value_column: str,
) -> float:
    if frame.empty or value_column not in frame.columns:
        return float("nan")

    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )

    weights = pd.to_numeric(
        frame.get(
            "MIN",
            pd.Series(
                1.0,
                index=frame.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    valid = (
        values.notna()
        & weights.gt(0)
    )

    if valid.any():
        return float(
            np.average(
                values.loc[valid],
                weights=weights.loc[valid],
            )
        )

    return float(
        values.mean()
    )


def _to_reference_percentile(
    value: float,
    reference: pd.Series,
) -> float:
    clean = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    if pd.isna(value) or clean.empty:
        return float("nan")

    return float(
        percentileofscore(
            clean,
            value,
            kind="rank",
        )
    )


def build_winning_context_sensitivity(
    scored_series: pd.DataFrame,
    scenarios: Mapping[
        str,
        Mapping[int, float],
    ] = ROUND_WEIGHT_SCENARIOS,
) -> pd.DataFrame:
    required = {
        "PLAYER_NAME",
        "ROUND_NUMBER",
        "TEAM_WON_SERIES",
        "EXPECTED_SERIES_WIN_PROB",
        "SERIES_OVERPERFORMANCE",
    }

    missing = required.difference(
        scored_series.columns
    )

    if missing:
        raise ValueError(
            "Scored playoff data is missing columns: "
            f"{sorted(missing)}"
        )

    frame = scored_series.dropna(
        subset=[
            "PLAYER_NAME",
        ]
    ).copy()

    if "SERIES_ID" in frame.columns:
        frame = frame.drop_duplicates(
            [
                "PLAYER_NAME",
                "SERIES_ID",
            ]
        )

    frame["ROUND_NUMBER"] = (
        pd.to_numeric(
            frame["ROUND_NUMBER"],
            errors="coerce",
        )
        .astype("Int64")
    )

    frame["TEAM_WON_SERIES"] = (
        pd.to_numeric(
            frame["TEAM_WON_SERIES"],
            errors="coerce",
        )
    )

    frame[
        "EXPECTED_SERIES_WIN_PROB"
    ] = pd.to_numeric(
        frame[
            "EXPECTED_SERIES_WIN_PROB"
        ],
        errors="coerce",
    )

    frame[
        "SERIES_OVERPERFORMANCE"
    ] = pd.to_numeric(
        frame[
            "SERIES_OVERPERFORMANCE"
        ],
        errors="coerce",
    )

    invalid_rounds = (
        frame["ROUND_NUMBER"]
        .dropna()
        .loc[
            lambda values:
            ~values.isin(
                [
                    1,
                    2,
                    3,
                    4,
                ]
            )
        ]
    )

    if not invalid_rounds.empty:
        raise ValueError(
            "Unexpected playoff round numbers: "
            f"{sorted(invalid_rounds.unique())}"
        )

    rows: list[
        dict[str, float | int | str]
    ] = []

    for scenario_name, round_weights in scenarios.items():
        weights = (
            frame["ROUND_NUMBER"]
            .map(round_weights)
            .astype(float)
        )

        if weights.isna().any():
            raise ValueError(
                "Round-weight scenario "
                f"{scenario_name!r} does not cover "
                "every playoff round."
            )

        scenario_frame = frame.copy()

        scenario_frame[
            "_ROUND_WEIGHT"
        ] = weights

        for player_name, group in scenario_frame.groupby(
            "PLAYER_NAME",
            sort=True,
        ):
            valid = (
                group[
                    "_ROUND_WEIGHT"
                ].notna()
                & group[
                    "SERIES_OVERPERFORMANCE"
                ].notna()
                & group[
                    "TEAM_WON_SERIES"
                ].notna()
                & group[
                    "EXPECTED_SERIES_WIN_PROB"
                ].notna()
            )

            group = group.loc[
                valid
            ].copy()

            if group.empty:
                continue

            total_weight = float(
                group[
                    "_ROUND_WEIGHT"
                ].sum()
            )

            weighted_actual_rate = float(
                np.average(
                    group[
                        "TEAM_WON_SERIES"
                    ],
                    weights=group[
                        "_ROUND_WEIGHT"
                    ],
                )
            )

            weighted_expected_rate = float(
                np.average(
                    group[
                        "EXPECTED_SERIES_WIN_PROB"
                    ],
                    weights=group[
                        "_ROUND_WEIGHT"
                    ],
                )
            )

            weighted_overperformance = (
                weighted_actual_rate
                - weighted_expected_rate
            )

            winning_context_score = float(
                np.clip(
                    50
                    + 50
                    * weighted_overperformance,
                    0,
                    100,
                )
            )

            rows.append(
                {
                    "PLAYER_NAME": str(
                        player_name
                    ),
                    "SCENARIO": (
                        scenario_name
                    ),
                    "SERIES": len(group),
                    "SERIES_WINS": int(
                        group[
                            "TEAM_WON_SERIES"
                        ].sum()
                    ),
                    "EXPECTED_WINS": float(
                        group[
                            "EXPECTED_SERIES_WIN_PROB"
                        ].sum()
                    ),
                    "WEIGHTED_SERIES_MASS": (
                        total_weight
                    ),
                    "WEIGHTED_ACTUAL_WIN_RATE": (
                        weighted_actual_rate
                    ),
                    "WEIGHTED_EXPECTED_WIN_RATE": (
                        weighted_expected_rate
                    ),
                    "WEIGHTED_OVERPERFORMANCE": (
                        weighted_overperformance
                    ),
                    "WINNING_CONTEXT_SCORE": (
                        winning_context_score
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "SCENARIO",
                "PLAYER_NAME",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def add_availability_adjusted_season_value(
    frame: pd.DataFrame,
    *,
    source_column: str = (
        "SEASON_VALUE_0_100"
    ),
    output_column: str = (
        "SEASON_VALUE_AVAILABILITY_ADJUSTED"
    ),
) -> pd.DataFrame:
    required = {
        "SEASON_TYPE",
        source_column,
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Season-value data is missing columns: "
            f"{sorted(missing)}"
        )

    result = frame.copy()

    base = pd.to_numeric(
        result[source_column],
        errors="coerce",
    )

    availability = pd.to_numeric(
        result.get(
            "availability",
            pd.Series(
                np.nan,
                index=result.index,
            ),
        ),
        errors="coerce",
    ).clip(
        0,
        1,
    )

    regular = _regular_season_mask(
        result["SEASON_TYPE"]
    )

    valid = (
        regular
        & base.notna()
        & availability.notna()
    )

    adjusted = base.copy()

    # Fifty is the model's average-season anchor. Availability
    # scales the contribution above or below that anchor rather
    # than treating missed games as zero-quality basketball.
    adjusted.loc[valid] = (
        50
        + (
            base.loc[valid]
            - 50
        )
        * availability.loc[valid]
    )

    result[
        output_column
    ] = adjusted.clip(
        0,
        100,
    )

    result[
        "AVAILABILITY_VALUE_SOURCE"
    ] = "quality_only"

    result.loc[
        valid,
        "AVAILABILITY_VALUE_SOURCE",
    ] = "availability_adjusted"

    result.loc[
        regular
        & base.notna()
        & availability.isna(),
        "AVAILABILITY_VALUE_SOURCE",
    ] = "missing_availability"

    return result


def _ensure_career_year(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    if "CAREER_YEAR" in result.columns:
        return result

    season_start = pd.to_numeric(
        result["SEASON"]
        .astype(str)
        .str[:4],
        errors="coerce",
    )

    player_ids = pd.to_numeric(
        result["PLAYER_ID"],
        errors="coerce",
    )

    first_year = season_start.groupby(
        player_ids
    ).transform(
        "min"
    )

    result["CAREER_YEAR"] = (
        season_start
        - first_year
        + 1
    )

    return result


def _build_value_reference(
    league_values: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    required = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "SEASON",
        "SEASON_TYPE",
        "MIN",
        value_column,
    }

    missing = required.difference(
        league_values.columns
    )

    if missing:
        raise ValueError(
            "League season values are missing columns: "
            f"{sorted(missing)}"
        )

    values = _ensure_career_year(
        league_values
    )

    regular = values[
        _regular_season_mask(
            values["SEASON_TYPE"]
        )
    ].copy()

    playoffs = values[
        values["SEASON_TYPE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("playoffs")
    ].copy()

    peak_summary = (
        summarize_peak_prime_longevity(
            values,
            value_column=value_column,
        )
    )

    rows: list[
        dict[str, float | int | str]
    ] = []

    group_columns = [
        "PLAYER_ID",
        "PLAYER_NAME",
    ]

    for (
        player_id,
        player_name,
    ), regular_group in regular.groupby(
        group_columns,
        sort=False,
    ):
        playoff_group = playoffs[
            pd.to_numeric(
                playoffs["PLAYER_ID"],
                errors="coerce",
            ).eq(
                float(player_id)
            )
        ]

        if "PLAYER_ID" in peak_summary.columns:
            peak_row = peak_summary[
                pd.to_numeric(
                    peak_summary[
                        "PLAYER_ID"
                    ],
                    errors="coerce",
                ).eq(
                    float(player_id)
                )
            ]
        else:
            peak_row = peak_summary[
                peak_summary[
                    "PLAYER_NAME"
                ].eq(
                    player_name
                )
            ]

        if peak_row.empty:
            continue

        peak_row = peak_row.iloc[0]

        top_regular = (
            pd.to_numeric(
                regular_group[
                    value_column
                ],
                errors="coerce",
            )
            .dropna()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        rows.append(
            {
                "PLAYER_ID": int(
                    player_id
                ),
                "PLAYER_NAME": str(
                    player_name
                ),
                "peak_raw": float(
                    peak_row[
                        "TOP_3_PEAK"
                    ]
                ),
                "prime_raw": float(
                    peak_row[
                        "BEST_7_CONSECUTIVE"
                    ]
                ),
                "longevity_raw": float(
                    peak_row[
                        "CAREER_VALUE_ABOVE_AVERAGE"
                    ]
                ),
                "regular_season_raw": (
                    float(
                        top_regular.mean()
                    )
                    if not top_regular.empty
                    else np.nan
                ),
                "playoffs_raw": (
                    _weighted_mean(
                        playoff_group,
                        value_column,
                    )
                ),
                "SEASONS": int(
                    regular_group[
                        "SEASON"
                    ].nunique()
                ),
                "TOTAL_MINUTES": float(
                    pd.to_numeric(
                        regular_group["MIN"],
                        errors="coerce",
                    ).sum()
                ),
            }
        )

    reference = pd.DataFrame(
        rows
    )

    return reference[
        reference["SEASONS"].ge(5)
        & reference[
            "TOTAL_MINUTES"
        ].ge(5000)
    ].reset_index(
        drop=True
    )


def build_availability_sensitivity(
    league_values: pd.DataFrame,
    target_ids: set[int],
) -> pd.DataFrame:
    adjusted = (
        add_availability_adjusted_season_value(
            league_values
        )
    )

    scenarios = {
        "quality_only": (
            league_values,
            "SEASON_VALUE_0_100",
        ),
        "availability_adjusted": (
            adjusted,
            (
                "SEASON_VALUE_"
                "AVAILABILITY_ADJUSTED"
            ),
        ),
    }

    rows: list[
        dict[str, float | int | str]
    ] = []

    for (
        scenario_name,
        (
            scenario_values,
            value_column,
        ),
    ) in scenarios.items():
        reference = _build_value_reference(
            scenario_values,
            value_column,
        )

        numeric_ids = pd.to_numeric(
            reference["PLAYER_ID"],
            errors="coerce",
        )

        targets = reference[
            numeric_ids.isin(
                target_ids
            )
        ].copy()

        for target_row in targets.itertuples(
            index=False
        ):
            for (
                category,
                raw_column,
            ) in (
                AVAILABILITY_SENSITIVE_CATEGORIES
                .items()
            ):
                raw_value = float(
                    getattr(
                        target_row,
                        raw_column,
                    )
                )

                score = (
                    _to_reference_percentile(
                        raw_value,
                        reference[
                            raw_column
                        ],
                    )
                )

                rows.append(
                    {
                        "PLAYER_ID": int(
                            target_row.PLAYER_ID
                        ),
                        "PLAYER_NAME": str(
                            target_row.PLAYER_NAME
                        ),
                        "SCENARIO": (
                            scenario_name
                        ),
                        "CATEGORY": category,
                        "RAW_VALUE": raw_value,
                        "SCORE": score,
                        "REFERENCE_SIZE": len(reference),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
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


def percentile_to_normal_score(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    probability = (
        numeric
        / 100
    ).clip(
        0.001,
        0.999,
    )

    normal_value = pd.Series(
        norm.ppf(
            probability
        ),
        index=values.index,
        dtype="float64",
    )

    anchor = float(
        norm.ppf(
            0.999
        )
    )

    transformed = (
        50
        + 50
        * normal_value
        / anchor
    )

    transformed = transformed.clip(
        0,
        100,
    )

    return transformed.where(
        numeric.notna()
    )


def apply_category_scale(
    category_scores: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    if scenario not in SCALE_SCENARIOS:
        raise ValueError(
            "Unknown category-scale scenario: "
            f"{scenario}"
        )

    result = category_scores.copy()

    if scenario == "historical_percentile":
        return result

    for category in (
        REFERENCE_PERCENTILE_CATEGORIES
    ):
        if category in result.columns:
            result[category] = (
                percentile_to_normal_score(
                    result[category]
                )
            )

    return result


def build_category_scaling_details(
    category_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, float | str]
    ] = []

    for scenario in SCALE_SCENARIOS:
        scaled = apply_category_scale(
            category_scores,
            scenario,
        )

        for row in scaled.itertuples(
            index=False
        ):
            for category in CATEGORIES:
                rows.append(
                    {
                        "SCENARIO": scenario,
                        "PLAYER_NAME": str(
                            row.PLAYER_NAME
                        ),
                        "CATEGORY": category,
                        "SCORE": float(
                            getattr(
                                row,
                                category,
                            )
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def _apply_availability_scenario(
    baseline: pd.DataFrame,
    availability_details: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    result = baseline.copy()

    if scenario == "quality_only":
        return result

    selected = availability_details[
        availability_details[
            "SCENARIO"
        ].eq(
            scenario
        )
    ]

    if selected.empty:
        raise ValueError(
            "No availability sensitivity rows "
            f"for scenario {scenario!r}."
        )

    for row in selected.itertuples(
        index=False
    ):
        mask = result[
            "PLAYER_NAME"
        ].eq(
            row.PLAYER_NAME
        )

        if not mask.any():
            raise ValueError(
                "Availability sensitivity contains "
                "an unknown player: "
                f"{row.PLAYER_NAME}"
            )

        result.loc[
            mask,
            row.CATEGORY,
        ] = float(
            row.SCORE
        )

    return result


def build_model_sensitivity_grid(
    baseline_category_scores: pd.DataFrame,
    availability_details: pd.DataFrame,
    winning_context_details: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    required = {
        "PLAYER_NAME",
        *CATEGORIES,
    }

    missing = required.difference(
        baseline_category_scores.columns
    )

    if missing:
        raise ValueError(
            "Category scores are missing columns: "
            f"{sorted(missing)}"
        )

    availability_scenarios = [
        "quality_only",
    ]

    availability_scenarios.extend(
        scenario
        for scenario in dict.fromkeys(
            availability_details[
                "SCENARIO"
            ].astype(str)
        )
        if scenario
        != "quality_only"
    )

    round_scenarios = list(
        dict.fromkeys(
            winning_context_details[
                "SCENARIO"
            ].astype(str)
        )
    )

    summary_rows: list[
        dict[str, float | int | str]
    ] = []

    driver_rows: list[
        dict[str, float | str]
    ] = []

    for availability_scenario in (
        availability_scenarios
    ):
        availability_frame = (
            _apply_availability_scenario(
                baseline_category_scores,
                availability_details,
                availability_scenario,
            )
        )

        for scale_scenario in SCALE_SCENARIOS:
            scaled = apply_category_scale(
                availability_frame,
                scale_scenario,
            )

            for round_scenario in round_scenarios:
                current = scaled.copy()

                winning = (
                    winning_context_details[
                        winning_context_details[
                            "SCENARIO"
                        ].eq(
                            round_scenario
                        )
                    ]
                    .set_index(
                        "PLAYER_NAME"
                    )[
                        "WINNING_CONTEXT_SCORE"
                    ]
                )

                current[
                    "winning_context"
                ] = current[
                    "PLAYER_NAME"
                ].map(
                    winning
                )

                numeric = current[
                    list(
                        CATEGORIES
                    )
                ].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                if numeric.isna().any().any():
                    missing_values = (
                        numeric.columns[
                            numeric.isna().any()
                        ].tolist()
                    )

                    raise ValueError(
                        "Sensitivity grid contains "
                        "missing category values: "
                        f"{missing_values}"
                    )

                current[
                    "EQUAL_WEIGHT_SCORE"
                ] = numeric.mean(
                    axis=1
                )

                current = (
                    current.sort_values(
                        "EQUAL_WEIGHT_SCORE",
                        ascending=False,
                    )
                    .reset_index(
                        drop=True
                    )
                )

                current["RANK"] = (
                    np.arange(
                        len(current)
                    )
                    + 1
                )

                total_score = float(
                    current[
                        "EQUAL_WEIGHT_SCORE"
                    ].sum()
                )

                for row in current.itertuples(
                    index=False
                ):
                    other_score = (
                        total_score
                        - float(
                            row.EQUAL_WEIGHT_SCORE
                        )
                        if len(current) == 2
                        else np.nan
                    )

                    summary_rows.append(
                        {
                            "AVAILABILITY_SCENARIO": (
                                availability_scenario
                            ),
                            "SCALE_SCENARIO": (
                                scale_scenario
                            ),
                            "ROUND_SCENARIO": (
                                round_scenario
                            ),
                            "PLAYER_NAME": str(
                                row.PLAYER_NAME
                            ),
                            "EQUAL_WEIGHT_SCORE": float(
                                row.EQUAL_WEIGHT_SCORE
                            ),
                            "RANK": int(
                                row.RANK
                            ),
                            "MARGIN_TO_OTHER": (
                                float(
                                    row.EQUAL_WEIGHT_SCORE
                                    - other_score
                                )
                                if len(current) == 2
                                else np.nan
                            ),
                        }
                    )

                if len(current) == 2:
                    ordered_players = sorted(
                        current[
                            "PLAYER_NAME"
                        ].astype(str)
                    )

                    player_a = ordered_players[0]
                    player_b = ordered_players[1]

                    indexed = current.set_index(
                        "PLAYER_NAME"
                    )

                    for category in CATEGORIES:
                        difference = float(
                            indexed.loc[
                                player_a,
                                category,
                            ]
                            - indexed.loc[
                                player_b,
                                category,
                            ]
                        )

                        driver_rows.append(
                            {
                                "AVAILABILITY_SCENARIO": (
                                    availability_scenario
                                ),
                                "SCALE_SCENARIO": (
                                    scale_scenario
                                ),
                                "ROUND_SCENARIO": (
                                    round_scenario
                                ),
                                "PLAYER_A": player_a,
                                "PLAYER_B": player_b,
                                "CATEGORY": category,
                                "PLAYER_A_MINUS_B": (
                                    difference
                                ),
                                "EQUAL_WEIGHT_CONTRIBUTION": (
                                    difference
                                    / len(
                                        CATEGORIES
                                    )
                                ),
                            }
                        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "AVAILABILITY_SCENARIO",
                "SCALE_SCENARIO",
                "ROUND_SCENARIO",
                "RANK",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    drivers = (
        pd.DataFrame(
            driver_rows
        )
        .sort_values(
            [
                "AVAILABILITY_SCENARIO",
                "SCALE_SCENARIO",
                "ROUND_SCENARIO",
                "CATEGORY",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary, drivers
