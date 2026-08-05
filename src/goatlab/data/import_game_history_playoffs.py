from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from goatlab.settings import settings
from goatlab.utils import write_parquet

SOURCE_DIR = Path("data/external/nba_game_history")

START_END_YEAR = 1985
END_END_YEAR = 1996


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def read_csv(filename: str) -> pd.DataFrame:
    path = SOURCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required historical file: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def to_numeric(
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


def load_team_possessions() -> pd.DataFrame:
    teams = read_csv("TeamStatistics.csv")

    teams = teams[
        teams["gameType"]
        .astype(str)
        .str.casefold()
        .eq("playoffs")
    ].copy()

    numeric_columns = [
        "gameId",
        "teamId",
        "numMinutes",
        "fieldGoalsAttempted",
        "freeThrowsAttempted",
        "reboundsOffensive",
        "turnovers",
    ]

    teams = to_numeric(
        teams,
        numeric_columns,
    )

    teams = teams.dropna(
        subset=[
            "gameId",
            "teamId",
        ]
    )

    # Standard box-score possession estimate.
    teams["TEAM_POSS_EST"] = (
        teams["fieldGoalsAttempted"].fillna(0)
        + 0.44
        * teams["freeThrowsAttempted"].fillna(0)
        - teams["reboundsOffensive"].fillna(0)
        + teams["turnovers"].fillna(0)
    )

    # Average both teams' estimates to produce one game pace value.
    game_possessions = (
        teams.groupby("gameId")["TEAM_POSS_EST"]
        .mean()
        .rename("GAME_POSS_EST")
        .reset_index()
    )

    teams = teams.merge(
        game_possessions,
        on="gameId",
        how="left",
        validate="many_to_one",
    )

    teams = teams.rename(
        columns={
            "teamId": "playerteamId",
            "numMinutes": "TEAM_MINUTES",
        }
    )

    return teams[
        [
            "gameId",
            "playerteamId",
            "TEAM_MINUTES",
            "GAME_POSS_EST",
        ]
    ].drop_duplicates(
        [
            "gameId",
            "playerteamId",
        ]
    )


def load_player_games() -> pd.DataFrame:
    players = read_csv("PlayerStatistics.csv")

    players = players[
        players["gameType"]
        .astype(str)
        .str.casefold()
        .eq("playoffs")
    ].copy()

    players["GAME_DATE"] = pd.to_datetime(
        players["gameDate"],
        errors="coerce",
    )

    players["SEASON_END_YEAR"] = (
        players["GAME_DATE"].dt.year
    )

    players = players[
        players["SEASON_END_YEAR"].between(
            START_END_YEAR,
            END_END_YEAR,
            inclusive="both",
        )
    ].copy()

    numeric_columns = [
        "personId",
        "gameId",
        "playerteamId",
        "numMinutes",
        "points",
        "assists",
        "blocks",
        "steals",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "threePointersAttempted",
        "threePointersMade",
        "freeThrowsAttempted",
        "freeThrowsMade",
        "reboundsDefensive",
        "reboundsOffensive",
        "reboundsTotal",
        "foulsPersonal",
        "turnovers",
    ]

    players = to_numeric(
        players,
        numeric_columns,
    )

    # Remove DNP/inactive rows.
    players = players[
        players["numMinutes"].fillna(0) > 0
    ].copy()

    players["PLAYER_NAME"] = (
        players["firstName"]
        .fillna("")
        .astype(str)
        .str.strip()
        + " "
        + players["lastName"]
        .fillna("")
        .astype(str)
        .str.strip()
    ).str.strip()

    teams = load_team_possessions()

    players = players.merge(
        teams,
        on=[
            "gameId",
            "playerteamId",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_possessions = players[
        players["GAME_POSS_EST"].isna()
    ]

    if not missing_possessions.empty:
        example = missing_possessions[
            [
                "gameId",
                "PLAYER_NAME",
                "playerteamId",
            ]
        ].head(20)

        raise ValueError(
            "Some player games could not be matched to team "
            "possession data:\n"
            + example.to_string(index=False)
        )

    # Estimated possessions while the player was on court.
    #
    # Team minutes are normally 240:
    # game possessions × player minutes × 5 / team minutes.
    players["PLAYER_POSS_EST"] = np.where(
        players["TEAM_MINUTES"] > 0,
        (
            players["GAME_POSS_EST"]
            * players["numMinutes"]
            * 5.0
            / players["TEAM_MINUTES"]
        ),
        np.nan,
    )

    return players


def aggregate_player_seasons(
    games: pd.DataFrame,
) -> pd.DataFrame:
    stat_mapping = {
        "numMinutes": "MIN",
        "points": "PTS",
        "assists": "AST",
        "blocks": "BLK",
        "steals": "STL",
        "fieldGoalsAttempted": "FGA",
        "fieldGoalsMade": "FGM",
        "threePointersAttempted": "FG3A",
        "threePointersMade": "FG3M",
        "freeThrowsAttempted": "FTA",
        "freeThrowsMade": "FTM",
        "reboundsDefensive": "DREB",
        "reboundsOffensive": "OREB",
        "reboundsTotal": "REB",
        "foulsPersonal": "PF",
        "turnovers": "TOV",
        "PLAYER_POSS_EST": "EST_POSS",
    }

    group_columns = [
        "personId",
        "PLAYER_NAME",
        "SEASON_END_YEAR",
    ]

    aggregation = {
        "gameId": pd.Series.nunique,
        **{
            source: "sum"
            for source in stat_mapping
        },
    }

    seasons = (
        games.groupby(
            group_columns,
            dropna=False,
        )
        .agg(aggregation)
        .reset_index()
        .rename(
            columns={
                "gameId": "GP",
                **stat_mapping,
            }
        )
    )

    team_names = (
        games.groupby(
            group_columns,
            dropna=False,
        )["playerteamName"]
        .agg(
            lambda values: "/".join(
                sorted(
                    {
                        str(value).strip()
                        for value in values
                        if pd.notna(value)
                    }
                )
            )
        )
        .rename("TEAM_ABBREVIATION")
        .reset_index()
    )

    seasons = seasons.merge(
        team_names,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    seasons["SEASON"] = (
        seasons["SEASON_END_YEAR"]
        .astype(int)
        .map(season_label)
    )

    seasons["PLAYER_ID"] = (
        seasons["personId"]
        .astype(int)
    )

    return seasons


def add_metadata(
    source: pd.DataFrame,
    measure_type: str,
    per_mode: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        index=source.index
    )

    output["PLAYER_ID"] = source["PLAYER_ID"]
    output["PLAYER_NAME"] = source["PLAYER_NAME"]
    output["TEAM_ABBREVIATION"] = source[
        "TEAM_ABBREVIATION"
    ]
    output["SEASON"] = source["SEASON"]
    output["SEASON_TYPE"] = "Playoffs"
    output["MEASURE_TYPE"] = measure_type
    output["PER_MODE"] = per_mode
    output["DATA_SOURCE"] = (
        "historical_nba_player_box_scores"
    )

    return output


def build_totals(
    seasons: pd.DataFrame,
) -> pd.DataFrame:
    output = add_metadata(
        seasons,
        measure_type="Base",
        per_mode="Totals",
    )

    stat_columns = [
        "GP",
        "MIN",
        "PTS",
        "AST",
        "BLK",
        "STL",
        "FGA",
        "FGM",
        "FG3A",
        "FG3M",
        "FTA",
        "FTM",
        "DREB",
        "OREB",
        "REB",
        "PF",
        "TOV",
    ]

    for column in stat_columns:
        output[column] = seasons[column]

    output["FG_PCT"] = np.where(
        output["FGA"] > 0,
        output["FGM"] / output["FGA"],
        np.nan,
    )

    output["FG3_PCT"] = np.where(
        output["FG3A"] > 0,
        output["FG3M"] / output["FG3A"],
        np.nan,
    )

    output["FT_PCT"] = np.where(
        output["FTA"] > 0,
        output["FTM"] / output["FTA"],
        np.nan,
    )

    output["EFG_PCT"] = np.where(
        output["FGA"] > 0,
        (
            output["FGM"]
            + 0.5 * output["FG3M"]
        )
        / output["FGA"],
        np.nan,
    )

    return output


def build_per100(
    seasons: pd.DataFrame,
) -> pd.DataFrame:
    output = add_metadata(
        seasons,
        measure_type="Base",
        per_mode="Per100Possessions",
    )

    output["GP"] = seasons["GP"]
    output["MIN"] = seasons["MIN"]

    rate_columns = [
        "PTS",
        "AST",
        "BLK",
        "STL",
        "FGA",
        "FGM",
        "FG3A",
        "FG3M",
        "FTA",
        "FTM",
        "DREB",
        "OREB",
        "REB",
        "PF",
        "TOV",
    ]

    denominator = seasons["EST_POSS"]

    for column in rate_columns:
        output[column] = np.where(
            denominator > 0,
            seasons[column] * 100.0 / denominator,
            np.nan,
        )

    output["FG_PCT"] = np.where(
        seasons["FGA"] > 0,
        seasons["FGM"] / seasons["FGA"],
        np.nan,
    )

    output["FG3_PCT"] = np.where(
        seasons["FG3A"] > 0,
        seasons["FG3M"] / seasons["FG3A"],
        np.nan,
    )

    output["FT_PCT"] = np.where(
        seasons["FTA"] > 0,
        seasons["FTM"] / seasons["FTA"],
        np.nan,
    )

    output["EFG_PCT"] = np.where(
        seasons["FGA"] > 0,
        (
            seasons["FGM"]
            + 0.5 * seasons["FG3M"]
        )
        / seasons["FGA"],
        np.nan,
    )

    return output


def build_historical_playoffs() -> pd.DataFrame:
    games = load_player_games()
    seasons = aggregate_player_seasons(games)

    totals = build_totals(seasons)
    per100 = build_per100(seasons)

    combined = pd.concat(
        [
            totals,
            per100,
        ],
        ignore_index=True,
        sort=False,
    )

    key_columns = [
        "PLAYER_ID",
        "SEASON",
        "SEASON_TYPE",
        "MEASURE_TYPE",
        "PER_MODE",
    ]

    duplicates = combined.duplicated(
        key_columns,
        keep=False,
    )

    if duplicates.any():
        example = combined.loc[
            duplicates,
            key_columns + ["PLAYER_NAME"],
        ]

        raise ValueError(
            "Duplicate historical playoff rows:\n"
            + example.head(30).to_string(index=False)
        )

    combined = combined.sort_values(
        [
            "SEASON",
            "PLAYER_NAME",
            "MEASURE_TYPE",
            "PER_MODE",
        ]
    ).reset_index(drop=True)

    output_path = (
        settings.interim_dir
        / "historical_playoff_seasons.parquet"
    )

    write_parquet(
        combined,
        output_path,
    )

    jordan = combined[
        (combined["PLAYER_ID"] == 893)
        & (combined["MEASURE_TYPE"] == "Base")
        & (combined["PER_MODE"] == "Totals")
    ].sort_values("SEASON")

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(combined):,}")
    print(
        "Coverage:",
        combined["SEASON"].min(),
        "through",
        combined["SEASON"].max(),
    )

    print("\nMichael Jordan historical playoff seasons:")
    print(
        jordan[
            [
                "SEASON",
                "GP",
                "MIN",
                "PTS",
                "REB",
                "AST",
            ]
        ].to_string(index=False)
    )

    return combined


if __name__ == "__main__":
    build_historical_playoffs()
