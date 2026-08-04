from __future__ import annotations

import numpy as np
import pandas as pd

from goatlab.models.integrity_sensitivity import (
    CATEGORIES,
    add_availability_adjusted_season_value,
    apply_category_scale,
    build_model_sensitivity_grid,
    build_winning_context_sensitivity,
)


def test_later_round_weights_reward_later_performance() -> None:
    rows = []

    for player_name in [
        "Player A",
        "Player B",
    ]:
        for round_number in [
            1,
            2,
            3,
            4,
        ]:
            overperformance = 0.0

            if (
                player_name == "Player A"
                and round_number == 4
            ):
                overperformance = 0.4

            if (
                player_name == "Player B"
                and round_number == 1
            ):
                overperformance = 0.4

            rows.append(
                {
                    "SERIES_ID": (
                        f"{player_name}-"
                        f"{round_number}"
                    ),
                    "PLAYER_NAME": player_name,
                    "ROUND_NUMBER": round_number,
                    "TEAM_WON_SERIES": (
                        1
                        if overperformance
                        else 0
                    ),
                    "EXPECTED_SERIES_WIN_PROB": (
                        0.6
                        if overperformance
                        else 0.0
                    ),
                    "SERIES_OVERPERFORMANCE": (
                        overperformance
                    ),
                }
            )

    result = (
        build_winning_context_sensitivity(
            pd.DataFrame(rows)
        )
        .set_index(
            [
                "SCENARIO",
                "PLAYER_NAME",
            ]
        )
    )

    equal_a = result.loc[
        (
            "equal_series",
            "Player A",
        ),
        "WINNING_CONTEXT_SCORE",
    ]

    equal_b = result.loc[
        (
            "equal_series",
            "Player B",
        ),
        "WINNING_CONTEXT_SCORE",
    ]

    assert np.isclose(
        equal_a,
        equal_b,
    )

    linear_a = result.loc[
        (
            "linear_round",
            "Player A",
        ),
        "WINNING_CONTEXT_SCORE",
    ]

    linear_b = result.loc[
        (
            "linear_round",
            "Player B",
        ),
        "WINNING_CONTEXT_SCORE",
    ]

    assert linear_a > linear_b


def test_availability_scales_distance_from_average() -> None:
    frame = pd.DataFrame(
        [
            {
                "SEASON_TYPE": "Regular Season",
                "SEASON_VALUE_0_100": 70.0,
                "availability": 0.5,
            },
            {
                "SEASON_TYPE": "Regular Season",
                "SEASON_VALUE_0_100": 40.0,
                "availability": 0.5,
            },
            {
                "SEASON_TYPE": "Playoffs",
                "SEASON_VALUE_0_100": 70.0,
                "availability": np.nan,
            },
            {
                "SEASON_TYPE": "Regular Season",
                "SEASON_VALUE_0_100": 70.0,
                "availability": np.nan,
            },
        ]
    )

    result = (
        add_availability_adjusted_season_value(
            frame
        )
    )

    values = result[
        "SEASON_VALUE_AVAILABILITY_ADJUSTED"
    ].tolist()

    assert np.allclose(
        values,
        [
            60.0,
            45.0,
            70.0,
            70.0,
        ],
    )


def _category_frame() -> pd.DataFrame:
    rows = []

    for player_name, base in [
        (
            "Player A",
            99.7,
        ),
        (
            "Player B",
            98.8,
        ),
    ]:
        row = {
            "PLAYER_NAME": player_name,
        }

        for category in CATEGORIES:
            row[category] = base

        row["winning_context"] = 60.0
        row["cultural_impact"] = 85.0

        rows.append(row)

    return pd.DataFrame(rows)


def test_normal_score_expands_top_tail_difference() -> None:
    baseline = _category_frame()

    transformed = (
        apply_category_scale(
            baseline,
            "normal_score_tail",
        )
        .set_index(
            "PLAYER_NAME"
        )
    )

    original = baseline.set_index(
        "PLAYER_NAME"
    )

    original_difference = (
        original.loc[
            "Player A",
            "peak",
        ]
        - original.loc[
            "Player B",
            "peak",
        ]
    )

    transformed_difference = (
        transformed.loc[
            "Player A",
            "peak",
        ]
        - transformed.loc[
            "Player B",
            "peak",
        ]
    )

    assert transformed_difference > (
        original_difference
    )

    assert (
        transformed.loc[
            "Player A",
            "peak",
        ]
        > transformed.loc[
            "Player B",
            "peak",
        ]
    )

    assert np.isclose(
        transformed.loc[
            "Player A",
            "winning_context",
        ],
        60.0,
    )


def test_sensitivity_grid_has_complete_combinations() -> None:
    baseline = _category_frame()

    availability_rows = []

    for player_name in [
        "Player A",
        "Player B",
    ]:
        for category in [
            "peak",
            "prime",
            "longevity",
            "regular_season",
            "playoffs",
        ]:
            availability_rows.append(
                {
                    "PLAYER_NAME": player_name,
                    "SCENARIO": (
                        "availability_adjusted"
                    ),
                    "CATEGORY": category,
                    "SCORE": (
                        95.0
                        if player_name
                        == "Player A"
                        else 94.0
                    ),
                }
            )

    winning = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player A",
                "SCENARIO": "equal_series",
                "WINNING_CONTEXT_SCORE": 60.0,
            },
            {
                "PLAYER_NAME": "Player B",
                "SCENARIO": "equal_series",
                "WINNING_CONTEXT_SCORE": 59.0,
            },
        ]
    )

    summary, drivers = (
        build_model_sensitivity_grid(
            baseline,
            pd.DataFrame(
                availability_rows
            ),
            winning,
        )
    )

    assert len(summary) == 8
    assert len(drivers) == 36

    assert not summary[
        "EQUAL_WEIGHT_SCORE"
    ].isna().any()

    assert set(
        summary[
            "AVAILABILITY_SCENARIO"
        ]
    ) == {
        "quality_only",
        "availability_adjusted",
    }

    assert set(
        summary[
            "SCALE_SCENARIO"
        ]
    ) == {
        "historical_percentile",
        "normal_score_tail",
    }
