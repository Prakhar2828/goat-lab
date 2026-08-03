from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamelog,
    leaguedashplayerstats,
    leaguedashteamstats,
    playerawards,
)

from goatlab.data.nba_client import CachedNbaClient
from goatlab.settings import settings
from goatlab.utils import load_yaml, season_range, write_parquet


SEASON_TYPES = ("Regular Season", "Playoffs")
MEASURE_TYPES = ("Base", "Advanced")
PER_MODES = ("Totals", "Per100Possessions")


def _first_frame(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not result:
        return pd.DataFrame()
    return next(iter(result.values())).copy()


def ingest_player_seasons(
    seasons: Iterable[str], client: CachedNbaClient
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        for season_type in SEASON_TYPES:
            for measure_type in MEASURE_TYPES:
                per_modes = PER_MODES if measure_type == "Base" else ("Totals",)
                for per_mode in per_modes:
                    result = client.fetch(
                        leaguedashplayerstats.LeagueDashPlayerStats,
                        "league_dash_player_stats",
                        season=season,
                        season_type_all_star=season_type,
                        measure_type_detailed_defense=measure_type,
                        per_mode_detailed=per_mode,
                    )
                    frame = _first_frame(result)
                    if frame.empty:
                        continue
                    frame["SEASON"] = season
                    frame["SEASON_TYPE"] = season_type
                    frame["MEASURE_TYPE"] = measure_type
                    frame["PER_MODE"] = per_mode
                    rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ingest_team_seasons(seasons: Iterable[str], client: CachedNbaClient) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        for season_type in SEASON_TYPES:
            for measure_type in MEASURE_TYPES:
                result = client.fetch(
                    leaguedashteamstats.LeagueDashTeamStats,
                    "league_dash_team_stats",
                    season=season,
                    season_type_all_star=season_type,
                    measure_type_detailed_defense=measure_type,
                    per_mode_detailed="Totals",
                )
                frame = _first_frame(result)
                if frame.empty:
                    continue
                frame["SEASON"] = season
                frame["SEASON_TYPE"] = season_type
                frame["MEASURE_TYPE"] = measure_type
                rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ingest_game_logs(seasons: Iterable[str], client: CachedNbaClient) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in seasons:
        for season_type in SEASON_TYPES:
            result = client.fetch(
                leaguegamelog.LeagueGameLog,
                "league_game_log",
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                sorter="DATE",
                direction="ASC",
            )
            frame = _first_frame(result)
            if frame.empty:
                continue
            frame["SEASON"] = season
            frame["SEASON_TYPE"] = season_type
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ingest_awards(client: CachedNbaClient) -> pd.DataFrame:
    source_config = load_yaml("configs/sources.yaml")
    rows: list[pd.DataFrame] = []
    for slug, player in source_config["players"].items():
        result = client.fetch(
            playerawards.PlayerAwards,
            "player_awards",
            player_id=player["player_id"],
        )
        frame = _first_frame(result)
        if frame.empty:
            continue
        frame["PLAYER_SLUG"] = slug
        frame["PLAYER_ID"] = player["player_id"]
        frame["PLAYER_NAME"] = player["display_name"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_core_ingestion(include_game_logs: bool = True) -> None:
    settings.ensure_directories()
    client = CachedNbaClient()
    seasons = season_range(settings.start_season, settings.end_season)

    player_seasons = ingest_player_seasons(seasons, client)
    write_parquet(player_seasons, settings.interim_dir / "league_player_seasons.parquet")

    team_seasons = ingest_team_seasons(seasons, client)
    write_parquet(team_seasons, settings.interim_dir / "league_team_seasons.parquet")

    if include_game_logs:
        game_logs = ingest_game_logs(seasons, client)
        write_parquet(game_logs, settings.interim_dir / "league_team_game_logs.parquet")

    awards = ingest_awards(client)
    write_parquet(awards, settings.interim_dir / "player_awards.parquet")
