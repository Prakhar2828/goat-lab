from __future__ import annotations

import numpy as np
import pandas as pd

from goatlab.features.era_adjust import (
    add_relative_metrics,
    add_true_shooting,
    shrink_z_scores,
)
from goatlab.features.availability import add_schedule_availability
from goatlab.settings import settings
from goatlab.utils import (
    load_yaml,
    read_optional_parquet,
    write_parquet,
)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        str(column).upper().strip()
        for column in result.columns
    ]
    return result


def _select_measure(
    frame: pd.DataFrame,
    measure_type: str,
    per_mode: str,
) -> pd.DataFrame:
    mask = (
        (frame["MEASURE_TYPE"] == measure_type)
        & (frame["PER_MODE"] == per_mode)
    )
    return frame.loc[mask].copy()


def build_player_feature_table() -> pd.DataFrame:
    nba_data = read_optional_parquet(
        settings.interim_dir / "league_player_seasons.parquet"
    )

    historical_data = read_optional_parquet(
        settings.interim_dir / "historical_player_seasons.parquet"
    )

    historical_playoffs = read_optional_parquet(
        settings.interim_dir / "historical_playoff_seasons.parquet"
    )

    if (
        nba_data.empty
        and historical_data.empty
        and historical_playoffs.empty
    ):
        raise FileNotFoundError(
            "Run NBA ingestion and historical imports "
            "before building features."
        )

    raw = pd.concat(
        [
            historical_data,
            historical_playoffs,
            nba_data,
        ],
        ignore_index=True,
        sort=False,
    )

    raw = _normalize_columns(raw)

    required_columns = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "SEASON",
        "SEASON_TYPE",
        "MEASURE_TYPE",
        "PER_MODE",
    }

    missing_columns = required_columns.difference(raw.columns)

    if missing_columns:
        raise ValueError(
            "Input player-season data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # NBA.com rows are placed after historical rows in the concat,
    # so keep="last" gives NBA.com precedence during overlap.
    raw = raw.drop_duplicates(
        subset=[
            "PLAYER_ID",
            "SEASON",
            "SEASON_TYPE",
            "MEASURE_TYPE",
            "PER_MODE",
        ],
        keep="last",
    ).reset_index(drop=True)

    base_totals = _select_measure(
        raw,
        measure_type="Base",
        per_mode="Totals",
    )

    base_per100 = _select_measure(
        raw,
        measure_type="Base",
        per_mode="Per100Possessions",
    )

    advanced = _select_measure(
        raw,
        measure_type="Advanced",
        per_mode="Totals",
    )

    keys = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "SEASON",
        "SEASON_TYPE",
    ]

    per100_keep = keys + [
        column
        for column in [
            "PTS",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
        ]
        if column in base_per100.columns
    ]

    per100 = base_per100[per100_keep].rename(
        columns={
            column: f"{column}_PER100"
            for column in per100_keep
            if column not in keys
        }
    )

    advanced_keep = keys + [
        column
        for column in [
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "AST_PCT",
            "AST_TO",
            "AST_RATIO",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "TM_TOV_PCT",
            "EFG_PCT",
            "TS_PCT",
            "USG_PCT",
            "PACE",
            "PIE",
            "PER",
            "OWS",
            "DWS",
            "WS",
            "WS_PER_48",
            "OBPM",
            "DBPM",
            "BPM",
            "VORP",
        ]
        if column in advanced.columns
    ]

    advanced = advanced[advanced_keep]

    features = base_totals.merge(
        per100,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    features = features.merge(
        advanced,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    features = add_true_shooting(features)

    if "TS_PCT" not in features.columns:
        features["TS_PCT"] = features["TS_PCT_CALC"]
    else:
        features["TS_PCT"] = features["TS_PCT"].fillna(
            features["TS_PCT_CALC"]
        )

    for stat in [
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
    ]:
        source = f"{stat}_PER100"

        if source in features.columns:
            features[f"{stat}_PER75"] = (
                pd.to_numeric(
                    features[source],
                    errors="coerce",
                )
                * 0.75
            )

    metrics = [
        column
        for column in [
            "PTS_PER75",
            "AST_PER75",
            "REB_PER75",
            "STL_PER75",
            "BLK_PER75",
            "TS_PCT",
            "NET_RATING",
            "OFF_RATING",
            "DEF_RATING",
            "AST_PCT",
            "REB_PCT",
            "USG_PCT",
            "PIE",
            "PER",
            "WS_PER_48",
            "OBPM",
            "DBPM",
            "BPM",
        ]
        if column in features.columns
    ]

    features = add_relative_metrics(
        features,
        group_columns=[
            "SEASON",
            "SEASON_TYPE",
        ],
        metric_columns=metrics,
        weight_column="MIN",
    )

    z_columns = [
        f"{metric}_Z"
        for metric in metrics
        if f"{metric}_Z" in features.columns
    ]

    features = shrink_z_scores(
        features,
        z_columns,
    )

    manual = read_optional_parquet(
        settings.interim_dir / "manual_advanced.parquet"
    )

    if not manual.empty:
        manual = _normalize_columns(manual)

        features = features.merge(
            manual,
            on=[
                "PLAYER_NAME",
                "SEASON",
                "SEASON_TYPE",
            ],
            how="left",
            suffixes=("", "_BREF"),
        )

    team_seasons = read_optional_parquet(
        settings.interim_dir
        / "league_team_seasons.parquet"
    )

    features = add_schedule_availability(
        features,
        team_seasons,
    )

    player_config = load_yaml(
        "configs/sources.yaml"
    )["players"]

    target_ids = {
        int(player["player_id"])
        for player in player_config.values()
    }

    numeric_player_ids = pd.to_numeric(
        features["PLAYER_ID"],
        errors="coerce",
    )

    target_features = features[
        numeric_player_ids.isin(target_ids)
    ].copy()

    target_features["PLAYER_ID"] = pd.to_numeric(
        target_features["PLAYER_ID"],
        errors="raise",
    ).astype(int)

    season_start_year = pd.to_numeric(
        target_features["SEASON"].str[:4],
        errors="coerce",
    )

    career_start_year = season_start_year.groupby(
        target_features["PLAYER_ID"]
    ).transform("min")

    target_features["CAREER_YEAR"] = (
        season_start_year
        - career_start_year
        + 1
    ).astype("Int64")

    gp = pd.to_numeric(
        target_features["GP"],
        errors="coerce",
    )

    is_regular_season = (
        target_features["SEASON_TYPE"]
        == "Regular Season"
    )

    # An 82-game denominator is valid only for regular seasons.
    # Playoff availability requires team playoff games and remains missing.

    write_parquet(
        features,
        settings.processed_dir
        / "league_player_features.parquet",
    )

    write_parquet(
        target_features,
        settings.processed_dir
        / "goat_player_features.parquet",
    )

    return target_features