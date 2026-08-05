from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

DEFAULT_SCHEDULE_GAMES = 82

SEASON_LENGTH_OVERRIDES: Mapping[
    str,
    int,
] = {
    "1998-99": 50,
    "2011-12": 66,
    "2020-21": 72,
}

TEAM_SPECIFIC_SEASONS = {
    "2019-20",
}


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


def _fallback_schedule_games(
    seasons: pd.Series,
) -> pd.Series:
    normalized = (
        seasons.fillna("")
        .astype(str)
        .str.strip()
    )

    fallback = normalized.map(
        SEASON_LENGTH_OVERRIDES
    )

    ordinary_season = (
        normalized.ne("")
        & ~normalized.isin(
            TEAM_SPECIFIC_SEASONS
        )
    )

    fallback = fallback.where(
        fallback.notna(),
        np.where(
            ordinary_season,
            DEFAULT_SCHEDULE_GAMES,
            np.nan,
        ),
    )

    return pd.to_numeric(
        fallback,
        errors="coerce",
    )


def build_team_schedule_table(
    team_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Return one actual regular-season GP value per team-season."""

    columns = [
        "SEASON",
        "TEAM_ID",
        "TEAM_GAMES_SCHEDULED",
    ]

    if team_seasons.empty:
        return pd.DataFrame(
            columns=columns
        )

    required = {
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ID",
        "GP",
    }

    missing = required.difference(
        team_seasons.columns
    )

    if missing:
        raise ValueError(
            "Team-season data is missing columns: "
            f"{sorted(missing)}"
        )

    frame = team_seasons[
        _regular_season_mask(
            team_seasons["SEASON_TYPE"]
        )
    ].copy()

    frame["SEASON"] = (
        frame["SEASON"]
        .astype(str)
        .str.strip()
    )

    frame["TEAM_ID"] = pd.to_numeric(
        frame["TEAM_ID"],
        errors="coerce",
    ).astype("Int64")

    frame["GP"] = pd.to_numeric(
        frame["GP"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "SEASON",
            "TEAM_ID",
            "GP",
        ]
    )

    schedule = (
        frame.groupby(
            [
                "SEASON",
                "TEAM_ID",
            ],
            as_index=False,
        )["GP"]
        .max()
        .rename(
            columns={
                "GP": (
                    "TEAM_GAMES_SCHEDULED"
                ),
            }
        )
    )

    invalid = (
        schedule[
            "TEAM_GAMES_SCHEDULED"
        ].le(0)
        | schedule[
            "TEAM_GAMES_SCHEDULED"
        ].gt(100)
    )

    if invalid.any():
        raise ValueError(
            "Invalid team schedule lengths:\n"
            + schedule.loc[
                invalid
            ].to_string(
                index=False
            )
        )

    return schedule


def add_schedule_availability(
    player_features: pd.DataFrame,
    team_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate player availability against actual team games.

    Team-season records are preferred. A fixed season-length rule is
    used only when a team record is unavailable, primarily for the
    historical pre-NBA-API portion of the dataset.
    """

    required = {
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ID",
        "GP",
    }

    missing = required.difference(
        player_features.columns
    )

    if missing:
        raise ValueError(
            "Player features are missing columns: "
            f"{sorted(missing)}"
        )

    result = player_features.copy()

    result["SEASON"] = (
        result["SEASON"]
        .astype(str)
        .str.strip()
    )

    result["_TEAM_ID_KEY"] = (
        pd.to_numeric(
            result["TEAM_ID"],
            errors="coerce",
        )
        .astype("Int64")
    )

    schedule = build_team_schedule_table(
        team_seasons
    ).rename(
        columns={
            "TEAM_ID": "_TEAM_ID_KEY",
        }
    )

    columns_to_remove = [
        column
        for column in [
            "TEAM_GAMES_SCHEDULED",
            "AVAILABILITY_SOURCE",
        ]
        if column in result.columns
    ]

    if columns_to_remove:
        result = result.drop(
            columns=columns_to_remove
        )

    result = result.merge(
        schedule,
        on=[
            "SEASON",
            "_TEAM_ID_KEY",
        ],
        how="left",
        validate="many_to_one",
    )

    regular = _regular_season_mask(
        result["SEASON_TYPE"]
    )

    fallback = _fallback_schedule_games(
        result["SEASON"]
    )

    team_schedule_available = (
        regular
        & result[
            "TEAM_GAMES_SCHEDULED"
        ].notna()
    )

    fallback_available = (
        regular
        & result[
            "TEAM_GAMES_SCHEDULED"
        ].isna()
        & fallback.notna()
    )

    result.loc[
        fallback_available,
        "TEAM_GAMES_SCHEDULED",
    ] = fallback.loc[
        fallback_available
    ]

    result[
        "AVAILABILITY_SOURCE"
    ] = "not_applicable"

    result.loc[
        team_schedule_available,
        "AVAILABILITY_SOURCE",
    ] = "team_season_table"

    result.loc[
        fallback_available,
        "AVAILABILITY_SOURCE",
    ] = "season_length_rule"

    missing_schedule = (
        regular
        & result[
            "TEAM_GAMES_SCHEDULED"
        ].isna()
    )

    result.loc[
        missing_schedule,
        "AVAILABILITY_SOURCE",
    ] = "missing"

    player_games = pd.to_numeric(
        result["GP"],
        errors="coerce",
    )

    denominator = pd.to_numeric(
        result["TEAM_GAMES_SCHEDULED"],
        errors="coerce",
    )

    valid = (
        regular
        & player_games.notna()
        & denominator.notna()
        & denominator.gt(0)
        & player_games.ge(0)
        & player_games.le(denominator)
    )

    result["availability"] = np.nan

    result.loc[
        valid,
        "availability",
    ] = (
        player_games.loc[valid]
        / denominator.loc[valid]
    )

    invalid_match = (
        regular
        & player_games.notna()
        & denominator.notna()
        & player_games.gt(denominator)
    )

    result.loc[
        invalid_match,
        "AVAILABILITY_SOURCE",
    ] = "invalid_team_match"

    return result.drop(
        columns=[
            "_TEAM_ID_KEY",
        ]
    )
