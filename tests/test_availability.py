from __future__ import annotations

import numpy as np
import pandas as pd

from goatlab.features.availability import (
    add_schedule_availability,
)


def test_shortened_seasons_use_team_games() -> None:
    player_features = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "LeBron James",
                "SEASON": "2011-12",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 1,
                "GP": 62,
            },
            {
                "PLAYER_NAME": "LeBron James",
                "SEASON": "2019-20",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 2,
                "GP": 67,
            },
            {
                "PLAYER_NAME": "LeBron James",
                "SEASON": "2020-21",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 2,
                "GP": 45,
            },
        ]
    )

    team_seasons = pd.DataFrame(
        [
            {
                "SEASON": "2011-12",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 1,
                "GP": 66,
                "MEASURE_TYPE": "Base",
            },
            {
                "SEASON": "2011-12",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 1,
                "GP": 66,
                "MEASURE_TYPE": "Advanced",
            },
            {
                "SEASON": "2019-20",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 2,
                "GP": 71,
                "MEASURE_TYPE": "Base",
            },
            {
                "SEASON": "2020-21",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 2,
                "GP": 72,
                "MEASURE_TYPE": "Base",
            },
        ]
    )

    result = add_schedule_availability(
        player_features,
        team_seasons,
    ).set_index("SEASON")

    assert np.isclose(
        result.loc[
            "2011-12",
            "availability",
        ],
        62 / 66,
    )

    assert np.isclose(
        result.loc[
            "2019-20",
            "availability",
        ],
        67 / 71,
    )

    assert np.isclose(
        result.loc[
            "2020-21",
            "availability",
        ],
        45 / 72,
    )

    assert (
        result[
            "AVAILABILITY_SOURCE"
        ]
        == "team_season_table"
    ).all()


def test_historical_season_uses_rule() -> None:
    player_features = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Michael Jordan",
                "SEASON": "1995-96",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 1,
                "GP": 82,
            },
        ]
    )

    result = add_schedule_availability(
        player_features,
        pd.DataFrame(),
    )

    assert np.isclose(
        result.loc[
            0,
            "availability",
        ],
        1.0,
    )

    assert (
        result.loc[
            0,
            "TEAM_GAMES_SCHEDULED",
        ]
        == 82
    )

    assert (
        result.loc[
            0,
            "AVAILABILITY_SOURCE",
        ]
        == "season_length_rule"
    )


def test_playoffs_have_no_availability() -> None:
    player_features = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "LeBron James",
                "SEASON": "2019-20",
                "SEASON_TYPE": "Playoffs",
                "TEAM_ID": 2,
                "GP": 21,
            },
        ]
    )

    team_seasons = pd.DataFrame(
        [
            {
                "SEASON": "2019-20",
                "SEASON_TYPE": "Regular Season",
                "TEAM_ID": 2,
                "GP": 71,
            },
        ]
    )

    result = add_schedule_availability(
        player_features,
        team_seasons,
    )

    assert pd.isna(
        result.loc[
            0,
            "availability",
        ]
    )

    assert (
        result.loc[
            0,
            "AVAILABILITY_SOURCE",
        ]
        == "not_applicable"
    )
