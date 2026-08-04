from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from goatlab.data.playoff_rounds import add_canonical_playoff_rounds
from goatlab.settings import settings


SOURCE_DIR = Path("data/external/nba_game_history")
START_SEASON_YEAR = 1984

TARGET_PLAYERS = {
    893: "Michael Jordan",
    2544: "LeBron James",
}


def _read_csv(
    filename: str,
    columns: list[str],
) -> pd.DataFrame:
    path = SOURCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required source file: {path}"
        )

    return pd.read_csv(
        path,
        usecols=columns,
        low_memory=False,
    )


def _add_season(
    frame: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Attach the NBA season containing each game.

    Regular-season games use the normal July boundary. Postseason
    games always belong to the season that began in the previous
    calendar year, including the delayed 2020 bubble playoffs and
    other postseason games played after June.
    """

    result = frame.copy()

    dates = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    regular_start_year = (
        dates.dt.year
        - dates.dt.month.lt(7).astype(
            "int64"
        )
    )

    start_year = regular_start_year

    if "gameType" in result.columns:
        game_type = (
            result["gameType"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )

        postseason = game_type.isin(
            {
                "playoffs",
                "play-in",
                "play-in tournament",
            }
        )

        postseason_start_year = (
            dates.dt.year - 1
        )

        start_year = start_year.where(
            ~postseason,
            postseason_start_year,
        )

    start_year = start_year.astype(
        "Int64"
    )

    end_year = (
        start_year + 1
    ).astype("Int64")

    result["GAME_DATE"] = dates
    result["SEASON_START_YEAR"] = (
        start_year
    )

    result["SEASON"] = (
        start_year.astype("string")
        + "-"
        + end_year.mod(100)
        .astype("string")
        .str.zfill(2)
    )

    return result

def _to_numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def _estimate_team_context(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    regular = team_games[
        team_games["gameType"]
        .astype(str)
        .str.casefold()
        .eq("regular season")
    ].copy()

    regular = _add_season(
        regular,
        "gameDate",
    )

    regular = regular[
        regular["SEASON_START_YEAR"]
        >= START_SEASON_YEAR
    ].copy()

    numeric_columns = [
        "gameId",
        "teamId",
        "opponentTeamId",
        "teamScore",
        "opponentScore",
        "fieldGoalsAttempted",
        "freeThrowsAttempted",
        "reboundsOffensive",
        "turnovers",
    ]

    regular = _to_numeric(
        regular,
        numeric_columns,
    )

    regular = regular.dropna(
        subset=[
            "gameId",
            "teamId",
            "opponentTeamId",
            "teamScore",
            "opponentScore",
            "SEASON",
        ]
    )

    regular["TEAM_ID"] = (
        regular["teamId"].astype("int64")
    )

    regular["OPP_TEAM_ID"] = (
        regular["opponentTeamId"]
        .astype("int64")
    )

    regular["MARGIN"] = (
        regular["teamScore"]
        - regular["opponentScore"]
    )

    regular["TEAM_POSS_EST"] = (
        regular[
            "fieldGoalsAttempted"
        ].fillna(0)
        + 0.44
        * regular[
            "freeThrowsAttempted"
        ].fillna(0)
        - regular[
            "reboundsOffensive"
        ].fillna(0)
        + regular["turnovers"].fillna(0)
    )

    regular = regular.drop_duplicates(
        [
            "SEASON",
            "gameId",
            "TEAM_ID",
        ]
    )

    aggregate = (
        regular.groupby(
            [
                "SEASON",
                "TEAM_ID",
            ],
            as_index=False,
        )
        .agg(
            GP=("gameId", "nunique"),
            POINTS_FOR=("teamScore", "sum"),
            POINTS_AGAINST=(
                "opponentScore",
                "sum",
            ),
            POSSESSIONS=(
                "TEAM_POSS_EST",
                "sum",
            ),
        )
    )

    aggregate["TEAM_NET_RATING"] = np.where(
        aggregate["POSSESSIONS"] > 0,
        (
            100.0
            * (
                aggregate["POINTS_FOR"]
                - aggregate["POINTS_AGAINST"]
            )
            / aggregate["POSSESSIONS"]
        ),
        np.nan,
    )

    srs_rows: list[
        dict[str, float | int | str]
    ] = []

    for season, group in regular.groupby(
        "SEASON"
    ):
        teams = sorted(
            group["TEAM_ID"].unique()
        )

        ratings = {
            int(team): 0.0
            for team in teams
        }

        for _ in range(100):
            updated: dict[int, float] = {}

            for team in teams:
                team_games_group = group[
                    group["TEAM_ID"] == team
                ]

                opponent_strength = (
                    team_games_group[
                        "OPP_TEAM_ID"
                    ]
                    .map(ratings)
                    .fillna(0.0)
                )

                updated[int(team)] = float(
                    (
                        team_games_group[
                            "MARGIN"
                        ]
                        + opponent_strength
                    ).mean()
                )

            center = float(
                np.mean(
                    list(updated.values())
                )
            )

            updated = {
                team: rating - center
                for team, rating
                in updated.items()
            }

            change = max(
                abs(
                    updated[team]
                    - ratings[team]
                )
                for team in teams
            )

            ratings = updated

            if change < 1e-8:
                break

        for team, rating in ratings.items():
            srs_rows.append(
                {
                    "SEASON": season,
                    "TEAM_ID": int(team),
                    "TEAM_SRS": float(
                        rating
                    ),
                }
            )

    srs = pd.DataFrame(srs_rows)

    context = aggregate.merge(
        srs,
        on=[
            "SEASON",
            "TEAM_ID",
        ],
        how="left",
        validate="one_to_one",
    )

    return context[
        [
            "SEASON",
            "TEAM_ID",
            "GP",
            "TEAM_SRS",
            "TEAM_NET_RATING",
        ]
    ]


def _build_seed_table(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    playoffs = team_games[
        team_games["gameType"]
        .astype(str)
        .str.casefold()
        .eq("playoffs")
    ].copy()

    playoffs = _add_season(
        playoffs,
        "gameDate",
    )

    playoffs = _to_numeric(
        playoffs,
        [
            "teamId",
            "seed",
        ],
    )

    playoffs = playoffs.dropna(
        subset=[
            "teamId",
            "SEASON",
        ]
    )

    playoffs["TEAM_ID"] = (
        playoffs["teamId"]
        .astype("int64")
    )

    seeds = (
        playoffs.groupby(
            [
                "SEASON",
                "TEAM_ID",
            ],
            as_index=False,
        )["seed"]
        .first()
        .rename(
            columns={
                "seed": "TEAM_SEED",
            }
        )
    )

    return seeds


def _build_target_team_map() -> dict[
    tuple[str, int],
    str,
]:
    players = _read_csv(
        "PlayerStatistics.csv",
        [
            "personId",
            "gameDate",
            "gameType",
            "playerteamId",
        ],
    )

    players = players[
        players["gameType"]
        .astype(str)
        .str.casefold()
        .eq("playoffs")
    ].copy()

    players = _to_numeric(
        players,
        [
            "personId",
            "playerteamId",
        ],
    )

    players = players[
        players["personId"].isin(
            TARGET_PLAYERS
        )
    ].copy()

    players = _add_season(
        players,
        "gameDate",
    )

    players = players.dropna(
        subset=[
            "personId",
            "playerteamId",
            "SEASON",
        ]
    )

    counts = (
        players.groupby(
            [
                "personId",
                "SEASON",
                "playerteamId",
            ],
            as_index=False,
        )
        .size()
        .sort_values(
            "size",
            ascending=False,
        )
        .drop_duplicates(
            [
                "personId",
                "SEASON",
            ]
        )
    )

    mapping: dict[
        tuple[str, int],
        str,
    ] = {}

    for row in counts.itertuples(
        index=False
    ):
        mapping[
            (
                str(row.SEASON),
                int(row.playerteamId),
            )
        ] = TARGET_PLAYERS[
            int(row.personId)
        ]

    return mapping


def build_playoff_series() -> pd.DataFrame:
    settings.ensure_directories()

    team_columns = [
        "gameId",
        "gameDate",
        "gameType",
        "teamId",
        "opponentTeamId",
        "teamScore",
        "opponentScore",
        "fieldGoalsAttempted",
        "freeThrowsAttempted",
        "reboundsOffensive",
        "turnovers",
        "seed",
    ]

    team_games = _read_csv(
        "TeamStatistics.csv",
        team_columns,
    )

    context = _estimate_team_context(
        team_games
    )

    seeds = _build_seed_table(
        team_games
    )

    target_map = _build_target_team_map()

    games = _read_csv(
        "Games.csv",
        [
            "gameId",
            "gameDate",
            "gameType",
            "hometeamId",
            "awayteamId",
            "winner",
            "gameLabel",
            "gameSubLabel",
            "seriesGameNumber",
        ],
    )

    games = games[
        games["gameType"]
        .astype(str)
        .str.casefold()
        .eq("playoffs")
    ].copy()

    games = _add_season(
        games,
        "gameDate",
    )

    games = games[
        games["SEASON_START_YEAR"]
        >= START_SEASON_YEAR
    ].copy()

    games = _to_numeric(
        games,
        [
            "gameId",
            "hometeamId",
            "awayteamId",
            "winner",
            "seriesGameNumber",
        ],
    )

    games = games.dropna(
        subset=[
            "gameId",
            "hometeamId",
            "awayteamId",
            "winner",
            "SEASON",
            "GAME_DATE",
        ]
    )

    games["HOME_TEAM_ID"] = (
        games["hometeamId"]
        .astype("int64")
    )

    games["AWAY_TEAM_ID"] = (
        games["awayteamId"]
        .astype("int64")
    )

    games["WINNER_TEAM_ID"] = (
        games["winner"]
        .astype("int64")
    )

    games["TEAM_LOW"] = games[
        [
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
        ]
    ].min(axis=1)

    games["TEAM_HIGH"] = games[
        [
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
        ]
    ].max(axis=1)

    games = games.drop_duplicates(
        "gameId"
    ).sort_values(
        [
            "SEASON",
            "GAME_DATE",
            "gameId",
        ]
    )

    context_lookup = context.set_index(
        [
            "SEASON",
            "TEAM_ID",
        ]
    )

    seed_lookup = seeds.set_index(
        [
            "SEASON",
            "TEAM_ID",
        ]
    )

    def context_value(
        season: str,
        team_id: int,
        column: str,
    ) -> float:
        key = (
            season,
            int(team_id),
        )

        if key not in context_lookup.index:
            return float("nan")

        return float(
            context_lookup.loc[
                key,
                column,
            ]
        )

    def seed_value(
        season: str,
        team_id: int,
    ) -> float:
        key = (
            season,
            int(team_id),
        )

        if key not in seed_lookup.index:
            return float("nan")

        value = seed_lookup.loc[
            key,
            "TEAM_SEED",
        ]

        return (
            float(value)
            if pd.notna(value)
            else float("nan")
        )

    rows: list[
        dict[str, object]
    ] = []

    grouping = [
        "SEASON",
        "TEAM_LOW",
        "TEAM_HIGH",
    ]

    for (
        season,
        team_low,
        team_high,
    ), series_games in games.groupby(
        grouping,
        sort=True,
    ):
        series_games = (
            series_games.sort_values(
                [
                    "GAME_DATE",
                    "gameId",
                ]
            )
            .reset_index(drop=True)
        )

        winner_counts = (
            series_games[
                "WINNER_TEAM_ID"
            ].value_counts()
        )

        if winner_counts.empty:
            continue

        series_winner = int(
            winner_counts.index[0]
        )

        team_low = int(team_low)
        team_high = int(team_high)

        game_one_home = int(
            series_games.iloc[0][
                "HOME_TEAM_ID"
            ]
        )

        round_name = (
            series_games["gameLabel"]
            .dropna()
            .astype(str)
            .iloc[0]
            if series_games[
                "gameLabel"
            ].notna().any()
            else ""
        )

        series_id = (
            f"{season}-"
            f"{team_low}-"
            f"{team_high}"
        )

        for team_id, opponent_id in [
            (
                team_low,
                team_high,
            ),
            (
                team_high,
                team_low,
            ),
        ]:
            team_wins = int(
                (
                    series_games[
                        "WINNER_TEAM_ID"
                    ]
                    == team_id
                ).sum()
            )

            opponent_wins = int(
                (
                    series_games[
                        "WINNER_TEAM_ID"
                    ]
                    == opponent_id
                ).sum()
            )

            rows.append(
                {
                    "SERIES_ID": series_id,
                    "PLAYER_NAME": (
                        target_map.get(
                            (
                                str(season),
                                int(team_id),
                            )
                        )
                    ),
                    "SEASON": str(season),
                    "ROUND": round_name,
                    "TEAM_ID": int(
                        team_id
                    ),
                    "OPP_TEAM_ID": int(
                        opponent_id
                    ),
                    "TEAM_WON_SERIES": int(
                        team_id
                        == series_winner
                    ),
                    "TEAM_SERIES_WINS": (
                        team_wins
                    ),
                    "OPP_SERIES_WINS": (
                        opponent_wins
                    ),
                    "SERIES_GAMES": int(
                        len(series_games)
                    ),
                    "HOME_COURT": int(
                        game_one_home
                        == team_id
                    ),
                    "TEAM_SRS": (
                        context_value(
                            str(season),
                            team_id,
                            "TEAM_SRS",
                        )
                    ),
                    "OPP_SRS": (
                        context_value(
                            str(season),
                            opponent_id,
                            "TEAM_SRS",
                        )
                    ),
                    "TEAM_NET_RATING": (
                        context_value(
                            str(season),
                            team_id,
                            "TEAM_NET_RATING",
                        )
                    ),
                    "OPP_NET_RATING": (
                        context_value(
                            str(season),
                            opponent_id,
                            "TEAM_NET_RATING",
                        )
                    ),
                    "TEAM_SEED": seed_value(
                        str(season),
                        team_id,
                    ),
                    "OPP_SEED": seed_value(
                        str(season),
                        opponent_id,
                    ),
                    "SERIES_START_DATE": (
                        series_games[
                            "GAME_DATE"
                        ].min()
                    ),
                    "SERIES_END_DATE": (
                        series_games[
                            "GAME_DATE"
                        ].max()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    result = add_canonical_playoff_rounds(
        result,
    )


    if result.empty:
        raise ValueError(
            "No playoff series were constructed."
        )

    # Avoid presenting entirely unavailable seed columns
    # as usable model features.
    for column in [
        "TEAM_SEED",
        "OPP_SEED",
    ]:
        if (
            column in result.columns
            and result[column]
            .notna()
            .sum()
            == 0
        ):
            result = result.drop(
                columns=column
            )

    result = result.sort_values(
        [
            "SEASON",
            "SERIES_ID",
            "TEAM_ID",
        ]
    ).reset_index(drop=True)

    output_path = (
        settings.manual_dir
        / "playoff_series.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    targets = result[
        result["PLAYER_NAME"].notna()
    ].copy()

    print(f"Wrote: {output_path}")
    print(
        f"Team-perspective rows: "
        f"{len(result):,}"
    )
    print(
        f"Unique playoff series: "
        f"{result['SERIES_ID'].nunique():,}"
    )

    print("\nModel-feature coverage:")
    print(
        result[
            [
                column
                for column in [
                    "TEAM_SRS",
                    "OPP_SRS",
                    "TEAM_NET_RATING",
                    "OPP_NET_RATING",
                    "TEAM_SEED",
                    "OPP_SEED",
                    "HOME_COURT",
                ]
                if column
                in result.columns
            ]
        ]
        .notna()
        .sum()
        .to_string()
    )

    print("\nJordan and LeBron series:")
    print(
        targets.groupby(
            "PLAYER_NAME"
        )
        .agg(
            SERIES=("SERIES_ID", "nunique"),
            WINS=("TEAM_WON_SERIES", "sum"),
            FIRST_SEASON=("SEASON", "min"),
            LAST_SEASON=("SEASON", "max"),
        )
        .to_string()
    )

    return result


if __name__ == "__main__":
    build_playoff_series()
