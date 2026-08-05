from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from goatlab.reporting.category_scores import (
    build_category_scores,
)

PROCESSED_DIR = Path(
    "data/processed"
)

TARGET_PLAYERS = [
    "Michael Jordan",
    "LeBron James",
]

SHORTENED_SEASONS = [
    "1998-99",
    "2011-12",
    "2019-20",
    "2020-21",
]


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def audit_category_scores() -> None:
    section(
        "CATEGORY SCORE COMPRESSION"
    )

    scores = build_category_scores()

    print(
        scores.to_string(
            index=False
        )
    )

    indexed = scores.set_index(
        "PLAYER_NAME"
    )

    missing_players = (
        set(TARGET_PLAYERS)
        - set(indexed.index)
    )

    if missing_players:
        raise ValueError(
            "Missing target players: "
            f"{sorted(missing_players)}"
        )

    jordan = indexed.loc[
        "Michael Jordan"
    ]

    lebron = indexed.loc[
        "LeBron James"
    ]

    differences = pd.DataFrame(
        {
            "CATEGORY": scores.columns[
                scores.columns
                != "PLAYER_NAME"
            ],
        }
    )

    differences[
        "LEBRON_MINUS_JORDAN"
    ] = [
        float(
            lebron[column]
            - jordan[column]
        )
        for column in differences[
            "CATEGORY"
        ]
    ]

    differences[
        "ABSOLUTE_DIFFERENCE"
    ] = differences[
        "LEBRON_MINUS_JORDAN"
    ].abs()

    differences[
        "CATEGORY_MEAN"
    ] = [
        float(
            (
                lebron[column]
                + jordan[column]
            )
            / 2
        )
        for column in differences[
            "CATEGORY"
        ]
    ]

    differences = differences.sort_values(
        "ABSOLUTE_DIFFERENCE",
        ascending=False,
    )

    print(
        "\nCategory differences:"
    )

    print(
        differences.to_string(
            index=False
        )
    )

    saturated = differences[
        differences[
            "CATEGORY_MEAN"
        ] >= 95
    ]

    print(
        "\nCategories with a "
        "two-player mean of at least 95:"
    )

    if saturated.empty:
        print("None")
    else:
        print(
            saturated[
                [
                    "CATEGORY",
                    "CATEGORY_MEAN",
                    "ABSOLUTE_DIFFERENCE",
                ]
            ].to_string(
                index=False
            )
        )


def audit_playoff_symmetry() -> None:
    section(
        "PLAYOFF PROBABILITY SYMMETRY"
    )

    path = (
        PROCESSED_DIR
        / "playoff_series_scored.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing playoff file: {path}"
        )

    frame = pd.read_parquet(
        path
    )

    required = {
        "SERIES_ID",
        "SEASON",
        "TEAM_WON_SERIES",
        "EXPECTED_SERIES_WIN_PROB",
        "SERIES_OVERPERFORMANCE",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Playoff file is missing: "
            f"{sorted(missing)}"
        )

    print(
        "Rows:",
        len(frame),
    )

    print(
        "Series:",
        frame["SERIES_ID"].nunique(),
    )

    print(
        "Seasons:",
        frame["SEASON"].nunique(),
    )

    series_sizes = (
        frame.groupby(
            "SERIES_ID"
        )
        .size()
    )

    winner_sums = (
        frame.groupby(
            "SERIES_ID"
        )[
            "TEAM_WON_SERIES"
        ]
        .sum()
    )

    probability_sums = (
        frame.groupby(
            "SERIES_ID"
        )[
            "EXPECTED_SERIES_WIN_PROB"
        ]
        .sum()
    )

    symmetry_error = (
        probability_sums
        - 1.0
    ).abs()

    print(
        "\nRows per series:"
    )

    print(
        series_sizes.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nWinners per series:"
    )

    print(
        winner_sums.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nProbability-sum statistics:"
    )

    print(
        probability_sums.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print(
        "\nAbsolute symmetry-error statistics:"
    )

    print(
        symmetry_error.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print(
        "\nSeries with symmetry error > 0.01:",
        int(
            (
                symmetry_error
                > 0.01
            ).sum()
        ),
    )

    print(
        "Series with symmetry error > 0.05:",
        int(
            (
                symmetry_error
                > 0.05
            ).sum()
        ),
    )

    worst = (
        symmetry_error.sort_values(
            ascending=False
        )
        .head(10)
        .rename(
            "ABS_SYMMETRY_ERROR"
        )
        .reset_index()
    )

    worst = worst.merge(
        probability_sums.rename(
            "PROBABILITY_SUM"
        ).reset_index(),
        on="SERIES_ID",
        how="left",
    )

    print(
        "\nTen largest symmetry errors:"
    )

    print(
        worst.to_string(
            index=False
        )
    )

    round_candidates = [
        "ROUND",
        "ROUND_NUMBER",
        "SERIES_ROUND",
        "PLAYOFF_ROUND",
    ]

    round_column = next(
        (
            column
            for column
            in round_candidates
            if column in frame.columns
        ),
        None,
    )

    print(
        "\nRound column:",
        round_column,
    )

    if round_column is not None:
        print(
            "\nRows by round:"
        )

        print(
            frame[
                round_column
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

        targets = frame[
            frame[
                "PLAYER_NAME"
            ].isin(
                TARGET_PLAYERS
            )
        ].copy()

        if not targets.empty:
            by_round = (
                targets.groupby(
                    [
                        "PLAYER_NAME",
                        round_column,
                    ],
                    as_index=False,
                )
                .agg(
                    SERIES=(
                        "SERIES_ID",
                        "nunique",
                    ),
                    SERIES_WINS=(
                        "TEAM_WON_SERIES",
                        "sum",
                    ),
                    EXPECTED_WINS=(
                        "EXPECTED_SERIES_WIN_PROB",
                        "sum",
                    ),
                    MEAN_OVERPERFORMANCE=(
                        "SERIES_OVERPERFORMANCE",
                        "mean",
                    ),
                )
            )

            print(
                "\nTarget performance by round:"
            )

            print(
                by_round.to_string(
                    index=False
                )
            )

    assert (
        series_sizes == 2
    ).all()

    assert (
        winner_sums == 1
    ).all()

    assert frame[
        "EXPECTED_SERIES_WIN_PROB"
    ].between(
        0,
        1,
    ).all()


def audit_availability_inputs() -> None:
    section(
        "AVAILABILITY AND SHORTENED SEASONS"
    )

    candidates: list[
        tuple[
            Path,
            pd.DataFrame,
        ]
    ] = []

    for path in sorted(
        PROCESSED_DIR.glob(
            "*.parquet"
        )
    ):
        try:
            frame = pd.read_parquet(
                path
            )
        except Exception as exc:
            print(
                f"Skipped {path.name}: {exc}"
            )
            continue

        required = {
            "PLAYER_NAME",
            "SEASON",
            "GP",
        }

        if required.issubset(
            frame.columns
        ):
            candidates.append(
                (
                    path,
                    frame,
                )
            )

    if not candidates:
        print(
            "No processed parquet contains "
            "PLAYER_NAME, SEASON, and GP."
        )
        return

    availability_names = {
        "AVAILABILITY",
        "availability",
        "AVAILABILITY_RATE",
        "availability_rate",
        "GAMES_AVAILABLE",
        "TEAM_GAMES",
        "LEAGUE_GAMES",
        "SCHEDULE_GAMES",
        "SEASON_GAMES",
    }

    for path, frame in candidates:
        print()
        print(
            f"Candidate: {path}"
        )

        relevant_columns = [
            column
            for column in frame.columns
            if (
                column in availability_names
                or column
                in {
                    "PLAYER_NAME",
                    "SEASON",
                    "TEAM_ID",
                    "TEAM_ABBREVIATION",
                    "GP",
                    "MIN",
                    "CAREER_YEAR",
                    "SEASON_VALUE",
                }
            )
        ]

        print(
            "Relevant columns:",
            relevant_columns,
        )

        target_rows = frame[
            frame[
                "PLAYER_NAME"
            ].isin(
                TARGET_PLAYERS
            )
        ].copy()

        shortened = target_rows[
            target_rows[
                "SEASON"
            ].isin(
                SHORTENED_SEASONS
            )
        ]

        if shortened.empty:
            print(
                "No target rows for shortened seasons."
            )
        else:
            print(
                "\nTarget shortened-season rows:"
            )

            print(
                shortened[
                    relevant_columns
                ].sort_values(
                    [
                        "PLAYER_NAME",
                        "SEASON",
                    ]
                ).to_string(
                    index=False
                )
            )

        for column in [
            "AVAILABILITY",
            "availability",
            "AVAILABILITY_RATE",
            "availability_rate",
        ]:
            if column not in target_rows.columns:
                continue

            numeric_availability = (
                pd.to_numeric(
                    target_rows[column],
                    errors="coerce",
                )
            )

            numeric_gp = pd.to_numeric(
                target_rows["GP"],
                errors="coerce",
            )

            implied_denominator = (
                numeric_gp
                / numeric_availability.replace(
                    0,
                    np.nan,
                )
            )

            implied = target_rows[
                [
                    "PLAYER_NAME",
                    "SEASON",
                    "GP",
                ]
            ].copy()

            implied[
                column
            ] = numeric_availability

            implied[
                "IMPLIED_GAMES_DENOMINATOR"
            ] = implied_denominator

            print(
                f"\nImplied denominator from {column}:"
            )

            print(
                implied[
                    implied[
                        "SEASON"
                    ].isin(
                        SHORTENED_SEASONS
                    )
                ].to_string(
                    index=False
                )
            )

        league_max = (
            frame.groupby(
                "SEASON",
                as_index=False,
            )["GP"]
            .max()
            .rename(
                columns={
                    "GP": "MAX_PLAYER_GP",
                }
            )
        )

        shortened_max = league_max[
            league_max[
                "SEASON"
            ].isin(
                SHORTENED_SEASONS
            )
        ]

        if not shortened_max.empty:
            print(
                "\nMaximum player GP by "
                "shortened season:"
            )

            print(
                shortened_max.to_string(
                    index=False
                )
            )


def list_processed_outputs() -> None:
    section(
        "PROCESSED OUTPUT INVENTORY"
    )

    for path in sorted(
        PROCESSED_DIR.glob(
            "*.parquet"
        )
    ):
        try:
            frame = pd.read_parquet(
                path
            )

            print(
                f"{path.name}: "
                f"{len(frame):,} rows, "
                f"{len(frame.columns)} columns"
            )

            print(
                "  "
                + ", ".join(
                    frame.columns
                )
            )
        except Exception as exc:
            print(
                f"{path.name}: ERROR: {exc}"
            )


def main() -> None:
    list_processed_outputs()
    audit_category_scores()
    audit_playoff_symmetry()
    audit_availability_inputs()

    section(
        "AUDIT COMPLETE"
    )

    print(
        "No production scoring logic "
        "was changed by this audit."
    )


if __name__ == "__main__":
    main()
