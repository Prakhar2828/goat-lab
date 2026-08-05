from __future__ import annotations

import re
import zlib
from pathlib import Path

import pandas as pd

from goatlab.settings import settings
from goatlab.utils import write_parquet

SOURCE_DIR = Path("data/external/bref_historical")

# Basketball-Reference uses the ending calendar year.
# 1985 means the 1984-85 NBA season.
START_END_YEAR = 1985
END_END_YEAR = 1996

TARGET_PLAYER_IDS = {
    "jordami01": 893,
    "jamesle01": 2544,
}


def season_label(end_year: int) -> str:
    start_year = end_year - 1
    return f"{start_year}-{str(end_year)[-2:]}"


def numeric_player_id(bref_id: str, player_name: str) -> int:
    bref_id = str(bref_id).strip()
    player_name = str(player_name).strip()

    if bref_id in TARGET_PLAYER_IDS:
        return TARGET_PLAYER_IDS[bref_id]

    # Name fallback in case the external ID changes.
    if player_name == "Michael Jordan":
        return 893

    if player_name == "LeBron James":
        return 2544

    # Assign every other historical player a stable negative ID.
    source = bref_id if bref_id and bref_id != "nan" else player_name
    checksum = zlib.crc32(source.encode("utf-8")) & 0x7FFFFFFF
    return -(checksum + 1)


def read_source(filename: str) -> pd.DataFrame:
    path = SOURCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing historical source: {path}")

    return pd.read_csv(path, low_memory=False)


def select_player_season_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one aggregate row per player and season.

    Basketball-Reference includes separate team rows for traded players,
    plus an aggregate row such as 2TM or 3TM. Prefer the aggregate row.
    """

    result = frame.copy()

    result["season"] = pd.to_numeric(
        result["season"],
        errors="coerce",
    ).astype("Int64")

    result["lg"] = (
        result["lg"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    result = result[
        (result["lg"] == "NBA")
        & result["season"].between(
            START_END_YEAR,
            END_END_YEAR,
            inclusive="both",
        )
    ].copy()

    result["player_id"] = (
        result["player_id"]
        .astype("string")
        .str.strip()
    )

    result["player"] = (
        result["player"]
        .astype("string")
        .str.strip()
    )

    result["team"] = (
        result["team"]
        .astype("string")
        .str.strip()
    )

    # Examples: 2TM, 3TM, 4TM.
    result["_aggregate_team"] = result["team"].str.fullmatch(
        re.compile(r"\d+TM"),
        na=False,
    )

    if "mp" in result.columns:
        result["_minutes_sort"] = pd.to_numeric(
            result["mp"],
            errors="coerce",
        ).fillna(-1)
    else:
        result["_minutes_sort"] = -1

    result = result.sort_values(
        [
            "season",
            "player_id",
            "_aggregate_team",
            "_minutes_sort",
        ],
        ascending=[True, True, False, False],
    )

    result = result.drop_duplicates(
        ["season", "player_id"],
        keep="first",
    )

    return result.drop(
        columns=["_aggregate_team", "_minutes_sort"],
    )


def create_metadata(
    source: pd.DataFrame,
    measure_type: str,
    per_mode: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=source.index)

    result["PLAYER_ID"] = [
        numeric_player_id(bref_id, player_name)
        for bref_id, player_name in zip(
            source["player_id"],
            source["player"],
            strict=False,
        )
    ]

    result["BREF_PLAYER_ID"] = source["player_id"].astype("string")
    result["PLAYER_NAME"] = source["player"].astype("string")
    result["TEAM_ABBREVIATION"] = source["team"].astype("string")

    result["SEASON"] = source["season"].astype(int).map(
        season_label
    )

    result["SEASON_TYPE"] = "Regular Season"
    result["MEASURE_TYPE"] = measure_type
    result["PER_MODE"] = per_mode
    result["DATA_SOURCE"] = "basketball_reference_kaggle"

    return result


def add_numeric_columns(
    output: pd.DataFrame,
    source: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    result = output.copy()

    for source_column, output_column in mapping.items():
        if source_column in source.columns:
            result[output_column] = pd.to_numeric(
                source[source_column],
                errors="coerce",
            )

    return result


def build_base_totals() -> pd.DataFrame:
    totals = select_player_season_rows(
        read_source("Player Totals.csv")
    )

    output = create_metadata(
        totals,
        measure_type="Base",
        per_mode="Totals",
    )

    mapping = {
        "g": "GP",
        "gs": "GS",
        "mp": "MIN",
        "fg": "FGM",
        "fga": "FGA",
        "fg_percent": "FG_PCT",
        "x3p": "FG3M",
        "x3pa": "FG3A",
        "x3p_percent": "FG3_PCT",
        "x2p": "FG2M",
        "x2pa": "FG2A",
        "x2p_percent": "FG2_PCT",
        "e_fg_percent": "EFG_PCT",
        "ft": "FTM",
        "fta": "FTA",
        "ft_percent": "FT_PCT",
        "orb": "OREB",
        "drb": "DREB",
        "trb": "REB",
        "ast": "AST",
        "stl": "STL",
        "blk": "BLK",
        "tov": "TOV",
        "pf": "PF",
        "pts": "PTS",
    }

    return add_numeric_columns(output, totals, mapping)


def build_base_per100() -> pd.DataFrame:
    per100 = select_player_season_rows(
        read_source("Per 100 Poss.csv")
    )

    output = create_metadata(
        per100,
        measure_type="Base",
        per_mode="Per100Possessions",
    )

    mapping = {
        "g": "GP",
        "gs": "GS",
        "mp": "MIN",
        "fg_per_100_poss": "FGM",
        "fga_per_100_poss": "FGA",
        "fg_percent": "FG_PCT",
        "x3p_per_100_poss": "FG3M",
        "x3pa_per_100_poss": "FG3A",
        "x3p_percent": "FG3_PCT",
        "x2p_per_100_poss": "FG2M",
        "x2pa_per_100_poss": "FG2A",
        "x2p_percent": "FG2_PCT",
        "e_fg_percent": "EFG_PCT",
        "ft_per_100_poss": "FTM",
        "fta_per_100_poss": "FTA",
        "ft_percent": "FT_PCT",
        "orb_per_100_poss": "OREB",
        "drb_per_100_poss": "DREB",
        "trb_per_100_poss": "REB",
        "ast_per_100_poss": "AST",
        "stl_per_100_poss": "STL",
        "blk_per_100_poss": "BLK",
        "tov_per_100_poss": "TOV",
        "pf_per_100_poss": "PF",
        "pts_per_100_poss": "PTS",
    }

    return add_numeric_columns(output, per100, mapping)


def build_advanced() -> pd.DataFrame:
    advanced = select_player_season_rows(
        read_source("Advanced.csv")
    )

    per100 = select_player_season_rows(
        read_source("Per 100 Poss.csv")
    )

    ratings = per100[
        [
            "season",
            "player_id",
            "o_rtg",
            "d_rtg",
        ]
    ].copy()

    advanced = advanced.merge(
        ratings,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )

    output = create_metadata(
        advanced,
        measure_type="Advanced",
        per_mode="Totals",
    )

    mapping = {
        "g": "GP",
        "gs": "GS",
        "mp": "MIN",
        "per": "PER",
        "ts_percent": "TS_PCT",
        "orb_percent": "OREB_PCT",
        "drb_percent": "DREB_PCT",
        "trb_percent": "REB_PCT",
        "ast_percent": "AST_PCT",
        "stl_percent": "STL_PCT",
        "blk_percent": "BLK_PCT",
        "tov_percent": "TM_TOV_PCT",
        "usg_percent": "USG_PCT",
        "ows": "OWS",
        "dws": "DWS",
        "ws": "WS",
        "ws_48": "WS_PER_48",
        "obpm": "OBPM",
        "dbpm": "DBPM",
        "bpm": "BPM",
        "vorp": "VORP",
        "o_rtg": "OFF_RATING",
        "d_rtg": "DEF_RATING",
    }

    output = add_numeric_columns(
        output,
        advanced,
        mapping,
    )

    output["NET_RATING"] = (
        output["OFF_RATING"]
        - output["DEF_RATING"]
    )

    return output


def build_historical_player_seasons() -> pd.DataFrame:
    totals = build_base_totals()
    per100 = build_base_per100()
    advanced = build_advanced()

    combined = pd.concat(
        [totals, per100, advanced],
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
            "Historical import contains duplicate player-season rows:\n"
            + example.head(20).to_string(index=False)
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
        / "historical_player_seasons.parquet"
    )

    write_parquet(combined, output_path)

    jordan = combined[
        (combined["PLAYER_ID"] == 893)
        & (combined["MEASURE_TYPE"] == "Base")
        & (combined["PER_MODE"] == "Totals")
    ]

    print(f"Wrote: {output_path}")
    print(f"Historical rows: {len(combined):,}")
    print(
        "Historical seasons:",
        combined["SEASON"].min(),
        "through",
        combined["SEASON"].max(),
    )
    print(
        "Michael Jordan seasons:",
        sorted(jordan["SEASON"].unique()),
    )

    return combined


if __name__ == "__main__":
    build_historical_player_seasons()
