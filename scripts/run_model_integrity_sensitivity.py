from __future__ import annotations

import pandas as pd

from goatlab.models.integrity_sensitivity import (
    build_availability_sensitivity,
    build_category_scaling_details,
    build_model_sensitivity_grid,
    build_winning_context_sensitivity,
)
from goatlab.settings import settings
from goatlab.utils import (
    load_yaml,
    write_parquet,
)

LINE = "=" * 100


def heading(
    title: str,
) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def main() -> None:
    category_path = (
        settings.processed_dir
        / "category_scores.parquet"
    )

    series_path = (
        settings.processed_dir
        / "playoff_series_scored.parquet"
    )

    values_path = (
        settings.processed_dir
        / "league_player_season_values.parquet"
    )

    for path in [
        category_path,
        series_path,
        values_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    category_scores = pd.read_parquet(
        category_path
    )

    scored_series = pd.read_parquet(
        series_path
    )

    league_values = pd.read_parquet(
        values_path
    )

    source_config = load_yaml(
        "configs/sources.yaml"
    )

    target_ids = {
        int(
            player[
                "player_id"
            ]
        )
        for player in source_config[
            "players"
        ].values()
    }

    winning = (
        build_winning_context_sensitivity(
            scored_series
        )
    )

    availability = (
        build_availability_sensitivity(
            league_values,
            target_ids,
        )
    )

    scaling = (
        build_category_scaling_details(
            category_scores
        )
    )

    summary, drivers = (
        build_model_sensitivity_grid(
            category_scores,
            availability,
            winning,
        )
    )

    write_parquet(
        winning,
        settings.processed_dir
        / "winning_context_sensitivity.parquet",
    )

    write_parquet(
        availability,
        settings.processed_dir
        / "availability_category_sensitivity.parquet",
    )

    write_parquet(
        scaling,
        settings.processed_dir
        / "category_scaling_sensitivity.parquet",
    )

    write_parquet(
        summary,
        settings.processed_dir
        / "model_integrity_sensitivity_grid.parquet",
    )

    write_parquet(
        drivers,
        settings.processed_dir
        / "model_integrity_sensitivity_drivers.parquet",
    )

    heading(
        "WINNING CONTEXT BY ROUND-WEIGHT SCENARIO"
    )

    print(
        winning[
            [
                "PLAYER_NAME",
                "SCENARIO",
                "SERIES",
                "SERIES_WINS",
                "EXPECTED_WINS",
                "WEIGHTED_OVERPERFORMANCE",
                "WINNING_CONTEXT_SCORE",
            ]
        ].to_string(
            index=False
        )
    )

    equal_context = (
        winning[
            winning[
                "SCENARIO"
            ].eq(
                "equal_series"
            )
        ]
        .set_index(
            "PLAYER_NAME"
        )[
            "WINNING_CONTEXT_SCORE"
        ]
    )

    baseline_context = (
        category_scores.set_index(
            "PLAYER_NAME"
        )[
            "winning_context"
        ]
    )

    common_players = (
        equal_context.index.intersection(
            baseline_context.index
        )
    )

    maximum_context_difference = float(
        (
            equal_context.loc[
                common_players
            ]
            - baseline_context.loc[
                common_players
            ]
        )
        .abs()
        .max()
    )

    print(
        "\nMaximum equal-series baseline difference:",
        maximum_context_difference,
    )

    assert (
        maximum_context_difference
        < 1e-8
    )

    heading(
        "AVAILABILITY CATEGORY SENSITIVITY"
    )

    availability_comparison = (
        availability.pivot_table(
            index=[
                "PLAYER_NAME",
                "CATEGORY",
            ],
            columns="SCENARIO",
            values=[
                "RAW_VALUE",
                "SCORE",
            ],
        )
    )

    print(
        availability_comparison.to_string()
    )

    heading(
        "MODEL SENSITIVITY GRID LEADERS"
    )

    leaders = summary[
        summary["RANK"].eq(1)
    ][
        [
            "AVAILABILITY_SCENARIO",
            "SCALE_SCENARIO",
            "ROUND_SCENARIO",
            "PLAYER_NAME",
            "EQUAL_WEIGHT_SCORE",
            "MARGIN_TO_OTHER",
        ]
    ]

    print(
        leaders.to_string(
            index=False
        )
    )

    heading(
        "LARGEST CATEGORY DRIVERS"
    )

    largest_drivers = (
        drivers.assign(
            ABS_CONTRIBUTION=lambda frame: (
                frame[
                    "EQUAL_WEIGHT_CONTRIBUTION"
                ].abs()
            )
        )
        .sort_values(
            "ABS_CONTRIBUTION",
            ascending=False,
        )
        .head(30)
    )

    print(
        largest_drivers[
            [
                "AVAILABILITY_SCENARIO",
                "SCALE_SCENARIO",
                "ROUND_SCENARIO",
                "CATEGORY",
                "PLAYER_A",
                "PLAYER_B",
                "PLAYER_A_MINUS_B",
                "EQUAL_WEIGHT_CONTRIBUTION",
            ]
        ].to_string(
            index=False
        )
    )

    heading(
        "SENSITIVITY SUMMARY"
    )

    leader_counts = (
        leaders[
            "PLAYER_NAME"
        ]
        .value_counts()
    )

    print(
        "Scenario count:",
        len(leaders),
    )

    print(
        "\nLeader counts:"
    )

    print(
        leader_counts.to_string()
    )

    print(
        "\nWrote:"
    )

    for filename in [
        "winning_context_sensitivity.parquet",
        "availability_category_sensitivity.parquet",
        "category_scaling_sensitivity.parquet",
        "model_integrity_sensitivity_grid.parquet",
        "model_integrity_sensitivity_drivers.parquet",
    ]:
        print(
            settings.processed_dir
            / filename
        )


if __name__ == "__main__":
    main()
