from __future__ import annotations

import numpy as np
import pandas as pd

from goatlab.settings import settings
from goatlab.utils import read_optional_parquet, write_parquet


def estimate_team_srs(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Approximate SRS from point differential and iterative schedule strength.

    The input is expected to contain one row per team-game. The iteration solves each
    team's rating as average margin plus average opponent rating.
    """
    required = {"GAME_ID", "TEAM_ID", "MATCHUP", "PTS"}
    if not required.issubset(game_logs.columns):
        return pd.DataFrame()

    logs = game_logs.copy()
    opponent = logs[["GAME_ID", "TEAM_ID", "PTS"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"}
    )
    paired = logs.merge(opponent, on="GAME_ID", how="inner")
    paired = paired[paired["TEAM_ID"] != paired["OPP_TEAM_ID"]]
    paired["MARGIN"] = paired["PTS"] - paired["OPP_PTS"]

    output: list[pd.DataFrame] = []
    grouping = [column for column in ["SEASON", "SEASON_TYPE"] if column in paired.columns]
    for keys, group in paired.groupby(grouping, dropna=False):
        teams = sorted(group["TEAM_ID"].unique())
        ratings = {team: 0.0 for team in teams}
        for _ in range(100):
            updated = {}
            for team in teams:
                team_games = group[group["TEAM_ID"] == team]
                margins = team_games["MARGIN"].astype(float)
                opponent_strength = team_games["OPP_TEAM_ID"].map(ratings).astype(float)
                updated[team] = float((margins + opponent_strength).mean())
            center = float(np.mean(list(updated.values())))
            updated = {team: value - center for team, value in updated.items()}
            if max(abs(updated[team] - ratings[team]) for team in teams) < 1e-8:
                ratings = updated
                break
            ratings = updated
        frame = pd.DataFrame({"TEAM_ID": teams, "SRS_EST": [ratings[team] for team in teams]})
        if grouping:
            key_values = keys if isinstance(keys, tuple) else (keys,)
            for column, value in zip(grouping, key_values, strict=False):
                frame[column] = value
        output.append(frame)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def build_team_context() -> pd.DataFrame:
    logs = read_optional_parquet(settings.interim_dir / "league_team_game_logs.parquet")
    if logs.empty:
        raise FileNotFoundError("Team game logs are missing. Run core ingestion with game logs.")
    logs.columns = [str(column).upper() for column in logs.columns]
    context = estimate_team_srs(logs)
    write_parquet(context, settings.processed_dir / "team_srs.parquet")
    return context
