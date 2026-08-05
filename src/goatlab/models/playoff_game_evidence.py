from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_PLAYERS = ("Michael Jordan", "LeBron James")

PLAYER_STATS_COLUMNS = (
    "firstName",
    "lastName",
    "personId",
    "gameId",
    "gameDate",
    "gameType",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
    "win",
    "home",
    "numMinutes",
    "points",
    "assists",
    "reboundsTotal",
    "reboundsOffensive",
    "reboundsDefensive",
    "steals",
    "blocks",
    "turnovers",
    "foulsPersonal",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "threePointersAttempted",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "plusMinusPoints",
    "playerteamId",
    "opponentteamId",
)

CORE_STAT_COLUMNS = (
    "MINUTES",
    "POINTS",
    "ASSISTS",
    "REBOUNDS",
    "STEALS",
    "BLOCKS",
    "TURNOVERS",
    "FGM",
    "FGA",
    "FTM",
    "FTA",
)

RELATIVE_METRICS = (
    "GAME_SCORE",
    "GAME_SCORE_PER36",
    "POINTS_PER36",
    "ASSISTS_PER36",
    "REBOUNDS_PER36",
    "STOCKS_PER36",
    "TURNOVERS_PER36",
)


@dataclass(frozen=True)
class PlayoffGamePolicy:
    minimum_baseline_minutes: float = 12.0
    bootstrap_repetitions: int = 5000
    bootstrap_seed: int = 23
    additional_central_weight: float = 0.0
    final_simulation_allowed: bool = False


def season_from_playoff_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    years = dates.dt.year.astype("Int64")
    previous = years - 1
    return previous.astype("string") + "-" + (years % 100).astype("string").str.zfill(2)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def normalize_player_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(PLAYER_STATS_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Player statistics are missing columns: {missing}")

    result = frame.loc[:, PLAYER_STATS_COLUMNS].copy()
    result["PLAYER_NAME"] = (
        result["firstName"].fillna("").astype(str).str.strip()
        + " "
        + result["lastName"].fillna("").astype(str).str.strip()
    ).str.strip()

    result["GAME_DATE"] = pd.to_datetime(result["gameDate"], errors="coerce")
    result["SEASON"] = season_from_playoff_date(result["GAME_DATE"])

    result = result.rename(
        columns={
            "personId": "PLAYER_ID",
            "gameId": "GAME_ID",
            "gameType": "GAME_TYPE",
            "gameLabel": "GAME_LABEL",
            "gameSubLabel": "GAME_SUB_LABEL",
            "seriesGameNumber": "SERIES_GAME_NUMBER",
            "win": "WIN",
            "home": "HOME",
            "numMinutes": "MINUTES",
            "points": "POINTS",
            "assists": "ASSISTS",
            "reboundsTotal": "REBOUNDS",
            "reboundsOffensive": "OFFENSIVE_REBOUNDS",
            "reboundsDefensive": "DEFENSIVE_REBOUNDS",
            "steals": "STEALS",
            "blocks": "BLOCKS",
            "turnovers": "TURNOVERS",
            "foulsPersonal": "PERSONAL_FOULS",
            "fieldGoalsMade": "FGM",
            "fieldGoalsAttempted": "FGA",
            "threePointersMade": "THREE_PM",
            "threePointersAttempted": "THREE_PA",
            "freeThrowsMade": "FTM",
            "freeThrowsAttempted": "FTA",
            "plusMinusPoints": "PLUS_MINUS",
            "playerteamId": "TEAM_ID",
            "opponentteamId": "OPP_TEAM_ID",
        }
    )

    numeric_columns = (
        "PLAYER_ID",
        "GAME_ID",
        "SERIES_GAME_NUMBER",
        "WIN",
        "HOME",
        "MINUTES",
        "POINTS",
        "ASSISTS",
        "REBOUNDS",
        "OFFENSIVE_REBOUNDS",
        "DEFENSIVE_REBOUNDS",
        "STEALS",
        "BLOCKS",
        "TURNOVERS",
        "PERSONAL_FOULS",
        "FGM",
        "FGA",
        "THREE_PM",
        "THREE_PA",
        "FTM",
        "FTA",
        "PLUS_MINUS",
        "TEAM_ID",
        "OPP_TEAM_ID",
    )
    result = _numeric(result, numeric_columns)

    result["GAME_TYPE"] = result["GAME_TYPE"].astype("string")
    return result


def add_game_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    minutes = pd.to_numeric(result["MINUTES"], errors="coerce")
    factor = 36.0 / minutes.where(minutes > 0)

    true_shooting_attempts = (
        pd.to_numeric(result["FGA"], errors="coerce")
        + 0.44 * pd.to_numeric(result["FTA"], errors="coerce")
    )
    result["TRUE_SHOOTING_ATTEMPTS"] = true_shooting_attempts
    result["TRUE_SHOOTING_PCT"] = (
        pd.to_numeric(result["POINTS"], errors="coerce")
        / (2.0 * true_shooting_attempts.where(true_shooting_attempts > 0))
    )

    result["POINTS_PER36"] = pd.to_numeric(result["POINTS"], errors="coerce") * factor
    result["ASSISTS_PER36"] = pd.to_numeric(result["ASSISTS"], errors="coerce") * factor
    result["REBOUNDS_PER36"] = pd.to_numeric(result["REBOUNDS"], errors="coerce") * factor
    result["STOCKS_PER36"] = (
        pd.to_numeric(result["STEALS"], errors="coerce")
        + pd.to_numeric(result["BLOCKS"], errors="coerce")
    ) * factor
    result["TURNOVERS_PER36"] = (
        pd.to_numeric(result["TURNOVERS"], errors="coerce") * factor
    )
    result["PLUS_MINUS_PER36"] = (
        pd.to_numeric(result["PLUS_MINUS"], errors="coerce") * factor
    )

    points = pd.to_numeric(result["POINTS"], errors="coerce")
    fgm = pd.to_numeric(result["FGM"], errors="coerce")
    fga = pd.to_numeric(result["FGA"], errors="coerce")
    ftm = pd.to_numeric(result["FTM"], errors="coerce")
    fta = pd.to_numeric(result["FTA"], errors="coerce")
    oreb = pd.to_numeric(result["OFFENSIVE_REBOUNDS"], errors="coerce")
    dreb = pd.to_numeric(result["DEFENSIVE_REBOUNDS"], errors="coerce")
    steals = pd.to_numeric(result["STEALS"], errors="coerce")
    assists = pd.to_numeric(result["ASSISTS"], errors="coerce")
    blocks = pd.to_numeric(result["BLOCKS"], errors="coerce")
    fouls = pd.to_numeric(result["PERSONAL_FOULS"], errors="coerce")
    turnovers = pd.to_numeric(result["TURNOVERS"], errors="coerce")

    result["GAME_SCORE"] = (
        points
        + 0.4 * fgm
        - 0.7 * fga
        - 0.4 * (fta - ftm)
        + 0.7 * oreb
        + 0.3 * dreb
        + steals
        + 0.7 * assists
        + 0.7 * blocks
        - 0.4 * fouls
        - turnovers
    )
    result["GAME_SCORE_PER36"] = result["GAME_SCORE"] * factor

    return result


def load_playoff_player_games(
    path: str | Path,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=list(PLAYER_STATS_COLUMNS),
        chunksize=chunksize,
        low_memory=False,
    ):
        normalized = normalize_player_statistics(chunk)
        playoffs = normalized[
            normalized["GAME_TYPE"].str.casefold().eq("playoffs")
        ].copy()
        if not playoffs.empty:
            pieces.append(playoffs)

    if not pieces:
        raise ValueError("No playoff player-game rows were found.")

    result = pd.concat(pieces, ignore_index=True)
    result = add_game_metrics(result)
    result = result.drop_duplicates(["PLAYER_ID", "GAME_ID"], keep="first")
    return result


def add_season_relative_metrics(
    playoff_games: pd.DataFrame,
    minimum_minutes: float = 12.0,
) -> pd.DataFrame:
    result = playoff_games.copy()
    eligible = pd.to_numeric(result["MINUTES"], errors="coerce").ge(minimum_minutes)

    for metric in RELATIVE_METRICS:
        values = pd.to_numeric(result[metric], errors="coerce")
        working = values.where(eligible)
        grouped = working.groupby(result["SEASON"])
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        result[f"{metric}_SEASON_Z"] = (values - mean) / std

    ts = pd.to_numeric(result["TRUE_SHOOTING_PCT"], errors="coerce")
    attempts = pd.to_numeric(result["TRUE_SHOOTING_ATTEMPTS"], errors="coerce")
    weighted_points = pd.to_numeric(result["POINTS"], errors="coerce")
    season_points = weighted_points.where(eligible).groupby(result["SEASON"]).transform("sum")
    season_attempts = attempts.where(eligible).groupby(result["SEASON"]).transform("sum")
    season_ts = season_points / (2.0 * season_attempts.where(season_attempts > 0))
    result["PLAYOFF_SEASON_TRUE_SHOOTING_PCT"] = season_ts
    result["TRUE_SHOOTING_PLUS"] = 100.0 * ts / season_ts.where(season_ts > 0)

    result["GAME_SCORE_PERCENTILE"] = np.nan
    eligible_values = result.loc[eligible, ["SEASON", "GAME_SCORE"]].copy()
    ranks = eligible_values.groupby("SEASON")["GAME_SCORE"].rank(
        method="average",
        pct=True,
    )
    result.loc[eligible_values.index, "GAME_SCORE_PERCENTILE"] = ranks

    return result


def match_candidate_games(
    playoff_games: pd.DataFrame,
    candidate_series: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "SERIES_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "OPP_TEAM_ID",
        "SERIES_START_DATE",
        "SERIES_END_DATE",
        "SERIES_GAMES",
        "TEAM_SERIES_WINS",
        "OPP_SERIES_WINS",
    }
    missing = sorted(required.difference(candidate_series.columns))
    if missing:
        raise ValueError(f"Candidate series are missing columns: {missing}")

    series = candidate_series.copy()
    series["SERIES_START_DATE"] = pd.to_datetime(
        series["SERIES_START_DATE"],
        errors="coerce",
    )
    series["SERIES_END_DATE"] = pd.to_datetime(
        series["SERIES_END_DATE"],
        errors="coerce",
    )

    targets = playoff_games[
        playoff_games["PLAYER_NAME"].isin(TARGET_PLAYERS)
    ].copy()

    matched: list[pd.DataFrame] = []
    mismatch_rows: list[dict[str, object]] = []

    for _, row in series.iterrows():
        games = targets[
            targets["PLAYER_NAME"].eq(row["PLAYER_NAME"])
            & targets["TEAM_ID"].eq(row["TEAM_ID"])
            & targets["OPP_TEAM_ID"].eq(row["OPP_TEAM_ID"])
            & targets["GAME_DATE"].between(
                row["SERIES_START_DATE"],
                row["SERIES_END_DATE"],
                inclusive="both",
            )
        ].copy()

        expected = int(row["SERIES_GAMES"])
        observed = int(games["GAME_ID"].nunique())
        if observed != expected:
            mismatch_rows.append(
                {
                    "SERIES_ID": row["SERIES_ID"],
                    "PLAYER_NAME": row["PLAYER_NAME"],
                    "EXPECTED_GAMES": expected,
                    "MATCHED_GAMES": observed,
                }
            )
            continue

        for column in series.columns:
            if column not in games.columns:
                games[column] = row[column]
        matched.append(games)

    if mismatch_rows:
        raise ValueError(
            "Candidate series game counts did not match: "
            f"{mismatch_rows[:10]}"
        )
    if not matched:
        raise ValueError("No candidate playoff games matched.")

    result = pd.concat(matched, ignore_index=True)
    duplicate_mask = result.duplicated(
        ["PLAYER_NAME", "GAME_ID"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = result.loc[
            duplicate_mask,
            ["PLAYER_NAME", "GAME_ID", "SERIES_ID"],
        ].to_dict("records")
        raise ValueError(f"Duplicate matched player-games: {duplicates[:10]}")

    return add_series_state_flags(result)


def add_series_state_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = result.sort_values(
        ["PLAYER_NAME", "SERIES_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    result["WIN"] = pd.to_numeric(result["WIN"], errors="coerce")
    grouped = result.groupby(["PLAYER_NAME", "SERIES_ID"], sort=False)

    result["TEAM_WINS_BEFORE"] = grouped["WIN"].cumsum() - result["WIN"]
    losses = 1.0 - result["WIN"]
    result["OPP_WINS_BEFORE"] = losses.groupby(
        [result["PLAYER_NAME"], result["SERIES_ID"]],
        sort=False,
    ).cumsum() - losses

    result["SERIES_TARGET_WINS"] = result[
        ["TEAM_SERIES_WINS", "OPP_SERIES_WINS"]
    ].max(axis=1)
    result["ELIMINATION_GAME"] = (
        result["OPP_WINS_BEFORE"]
        == result["SERIES_TARGET_WINS"] - 1
    )
    result["CLOSEOUT_OPPORTUNITY"] = (
        result["TEAM_WINS_BEFORE"]
        == result["SERIES_TARGET_WINS"] - 1
    )
    result["GAME_SEVEN"] = (
        pd.to_numeric(result["SERIES_GAME_NUMBER"], errors="coerce") == 7
    )
    result["SERIES_CLINCH_GAME"] = (
        result["CLOSEOUT_OPPORTUNITY"] & result["WIN"].eq(1)
    )
    return result


def _weighted_true_shooting(frame: pd.DataFrame) -> float:
    points = pd.to_numeric(frame["POINTS"], errors="coerce").sum(min_count=1)
    attempts = pd.to_numeric(
        frame["TRUE_SHOOTING_ATTEMPTS"],
        errors="coerce",
    ).sum(min_count=1)
    if pd.isna(points) or pd.isna(attempts) or attempts <= 0:
        return float("nan")
    return float(points / (2.0 * attempts))


def _weighted_ts_plus(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(frame["TRUE_SHOOTING_PLUS"], errors="coerce")
    weights = pd.to_numeric(
        frame["TRUE_SHOOTING_ATTEMPTS"],
        errors="coerce",
    )
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def summarize_series_games(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (player, series_id), group in frame.groupby(
        ["PLAYER_NAME", "SERIES_ID"],
        sort=True,
    ):
        rows.append(
            {
                "PLAYER_NAME": player,
                "SERIES_ID": series_id,
                "SEASON": group["SEASON"].iloc[0],
                "ROUND": group.get("ROUND", pd.Series([pd.NA])).iloc[0],
                "SERIES_GAMES": int(group["GAME_ID"].nunique()),
                "SERIES_WON": int(pd.to_numeric(group["TEAM_WON_SERIES"], errors="coerce").iloc[0]),
                "MINUTES_PER_GAME": float(pd.to_numeric(group["MINUTES"], errors="coerce").mean()),
                "POINTS_PER_GAME": float(pd.to_numeric(group["POINTS"], errors="coerce").mean()),
                "ASSISTS_PER_GAME": float(pd.to_numeric(group["ASSISTS"], errors="coerce").mean()),
                "REBOUNDS_PER_GAME": float(pd.to_numeric(group["REBOUNDS"], errors="coerce").mean()),
                "TRUE_SHOOTING_PCT": _weighted_true_shooting(group),
                "TRUE_SHOOTING_PLUS": _weighted_ts_plus(group),
                "GAME_SCORE": float(pd.to_numeric(group["GAME_SCORE"], errors="coerce").mean()),
                "GAME_SCORE_PER36": float(pd.to_numeric(group["GAME_SCORE_PER36"], errors="coerce").mean()),
                "GAME_SCORE_SEASON_Z": float(pd.to_numeric(group["GAME_SCORE_SEASON_Z"], errors="coerce").mean()),
                "GAME_SCORE_PERCENTILE": float(pd.to_numeric(group["GAME_SCORE_PERCENTILE"], errors="coerce").mean()),
                "PLUS_MINUS_PER_GAME": float(pd.to_numeric(group["PLUS_MINUS"], errors="coerce").mean()),
                "ELIMINATION_GAMES": int(group["ELIMINATION_GAME"].sum()),
                "CLOSEOUT_OPPORTUNITIES": int(group["CLOSEOUT_OPPORTUNITY"].sum()),
                "GAME_SEVENS": int(group["GAME_SEVEN"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_player_games(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player, group in frame.groupby("PLAYER_NAME", sort=True):
        elimination = group[group["ELIMINATION_GAME"]]
        closeout = group[group["CLOSEOUT_OPPORTUNITY"]]
        game_seven = group[group["GAME_SEVEN"]]
        rows.append(
            {
                "PLAYER_NAME": player,
                "GAMES": int(group["GAME_ID"].nunique()),
                "SERIES": int(group["SERIES_ID"].nunique()),
                "PLAYOFF_SEASONS": int(group["SEASON"].nunique()),
                "MINUTES_PER_GAME": float(pd.to_numeric(group["MINUTES"], errors="coerce").mean()),
                "POINTS_PER_GAME": float(pd.to_numeric(group["POINTS"], errors="coerce").mean()),
                "ASSISTS_PER_GAME": float(pd.to_numeric(group["ASSISTS"], errors="coerce").mean()),
                "REBOUNDS_PER_GAME": float(pd.to_numeric(group["REBOUNDS"], errors="coerce").mean()),
                "TRUE_SHOOTING_PCT": _weighted_true_shooting(group),
                "TRUE_SHOOTING_PLUS": _weighted_ts_plus(group),
                "GAME_SCORE": float(pd.to_numeric(group["GAME_SCORE"], errors="coerce").mean()),
                "GAME_SCORE_PER36": float(pd.to_numeric(group["GAME_SCORE_PER36"], errors="coerce").mean()),
                "GAME_SCORE_SEASON_Z": float(pd.to_numeric(group["GAME_SCORE_SEASON_Z"], errors="coerce").mean()),
                "MEDIAN_GAME_SCORE_PERCENTILE": float(pd.to_numeric(group["GAME_SCORE_PERCENTILE"], errors="coerce").median()),
                "POINTS_PER36": float(pd.to_numeric(group["POINTS_PER36"], errors="coerce").mean()),
                "ASSISTS_PER36": float(pd.to_numeric(group["ASSISTS_PER36"], errors="coerce").mean()),
                "REBOUNDS_PER36": float(pd.to_numeric(group["REBOUNDS_PER36"], errors="coerce").mean()),
                "STOCKS_PER36": float(pd.to_numeric(group["STOCKS_PER36"], errors="coerce").mean()),
                "TURNOVERS_PER36": float(pd.to_numeric(group["TURNOVERS_PER36"], errors="coerce").mean()),
                "PLUS_MINUS_PER_GAME": float(pd.to_numeric(group["PLUS_MINUS"], errors="coerce").mean()),
                "ELIMINATION_GAMES": int(len(elimination)),
                "ELIMINATION_GAME_SCORE": float(pd.to_numeric(elimination["GAME_SCORE"], errors="coerce").mean()),
                "ELIMINATION_WINS": int(pd.to_numeric(elimination["WIN"], errors="coerce").sum()),
                "CLOSEOUT_OPPORTUNITIES": int(len(closeout)),
                "CLOSEOUT_GAME_SCORE": float(pd.to_numeric(closeout["GAME_SCORE"], errors="coerce").mean()),
                "CLOSEOUT_WINS": int(pd.to_numeric(closeout["WIN"], errors="coerce").sum()),
                "GAME_SEVENS": int(len(game_seven)),
                "GAME_SEVEN_GAME_SCORE": float(pd.to_numeric(game_seven["GAME_SCORE"], errors="coerce").mean()),
                "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
                "PRIMARY_MODEL_ELIGIBLE": False,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_series_comparison(
    series_summary: pd.DataFrame,
    metrics: Iterable[str],
    repetitions: int = 5000,
    seed: int = 23,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = list(TARGET_PLAYERS)
    data = {
        player: series_summary[
            series_summary["PLAYER_NAME"].eq(player)
        ].reset_index(drop=True)
        for player in players
    }
    if any(frame.empty for frame in data.values()):
        raise ValueError("Both target players need series summaries.")

    rows: list[dict[str, object]] = []
    for metric in metrics:
        differences = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            player_means: dict[str, float] = {}
            for player in players:
                frame = data[player]
                sample_index = rng.integers(0, len(frame), size=len(frame))
                values = pd.to_numeric(
                    frame.iloc[sample_index][metric],
                    errors="coerce",
                )
                player_means[player] = float(values.mean())
            differences[index] = (
                player_means["LeBron James"]
                - player_means["Michael Jordan"]
            )

        observed = (
            pd.to_numeric(data["LeBron James"][metric], errors="coerce").mean()
            - pd.to_numeric(data["Michael Jordan"][metric], errors="coerce").mean()
        )
        rows.append(
            {
                "METRIC": metric,
                "LEBRON_MINUS_JORDAN": float(observed),
                "BOOTSTRAP_LOW_95": float(np.nanquantile(differences, 0.025)),
                "BOOTSTRAP_HIGH_95": float(np.nanquantile(differences, 0.975)),
                "PROBABILITY_LEBRON_GREATER": float(np.nanmean(differences > 0)),
                "BOOTSTRAP_REPETITIONS": int(repetitions),
                "CLUSTER_UNIT": "playoff_series",
                "PRIMARY_MODEL_ELIGIBLE": False,
                "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
            }
        )
    return pd.DataFrame(rows)


def core_stat_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player, group in frame.groupby("PLAYER_NAME", sort=True):
        for column in CORE_STAT_COLUMNS:
            observed = int(group[column].notna().sum())
            total = int(len(group))
            rows.append(
                {
                    "PLAYER_NAME": player,
                    "STAT": column,
                    "OBSERVATIONS": observed,
                    "TOTAL_GAMES": total,
                    "COVERAGE_RATE": observed / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)
