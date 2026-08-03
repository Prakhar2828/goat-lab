from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from goatlab.data.import_bref_historical import (
    numeric_player_id,
    season_label,
)
from goatlab.settings import settings
from goatlab.utils import write_parquet


START_END_YEAR = 1985
END_END_YEAR = 1996

SOURCE_DIR = Path("data/external/bref_historical")
CACHE_DIR = settings.raw_dir / "bref_playoffs"

REQUEST_DELAY_SECONDS = 4.0
REQUEST_TIMEOUT_SECONDS = 90
MAX_ATTEMPTS = 5

HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Referer": "https://www.basketball-reference.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
}

PAGE_TYPES = {
    "totals": {
        "url_suffix": "totals",
        "required_columns": {"Player", "G", "MP", "PTS"},
    },
    "per_poss": {
        "url_suffix": "per_poss",
        "required_columns": {
            "Player",
            "G",
            "MP",
            "PTS",
            "ORtg",
            "DRtg",
        },
    },
    "advanced": {
        "url_suffix": "advanced",
        "required_columns": {
            "Player",
            "G",
            "MP",
            "PER",
            "BPM",
            "VORP",
        },
    },
}


def clean_column_name(column: object) -> str:
    if isinstance(column, tuple):
        values = [
            str(value).strip()
            for value in column
            if str(value).strip()
            and not str(value).startswith("Unnamed")
            and str(value).lower() != "nan"
        ]

        if values:
            return values[-1]

    return str(column).strip()


def normalize_player_name(value: object) -> str:
    return str(value).strip().removesuffix("*").strip()


def normalize_team(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def download_html(
    session: requests.Session,
    url: str,
    cache_path: Path,
) -> str:
    if cache_path.exists():
        return cache_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            html = response.text

            if len(html.strip()) < 1_000:
                raise RuntimeError(
                    f"Response was unexpectedly short: "
                    f"{len(html)} characters"
                )

            blocked_markers = [
                "Rate Limit Exceeded",
                "Access Denied",
                "Attention Required",
            ]

            if any(marker in html for marker in blocked_markers):
                raise RuntimeError(
                    "Basketball-Reference returned a blocking page."
                )

            cache_path.write_text(
                html,
                encoding="utf-8",
            )

            time.sleep(REQUEST_DELAY_SECONDS)
            return html

        except Exception as exc:
            last_error = exc

            if attempt == MAX_ATTEMPTS:
                break

            wait_seconds = min(
                60,
                5 * (2 ** (attempt - 1)),
            )

            print(
                f"Attempt {attempt} failed for {url}: {exc}. "
                f"Waiting {wait_seconds} seconds."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Unable to download {url}"
    ) from last_error


def _cell_text(value: object) -> object:
    if isinstance(value, tuple):
        return value[0]

    return value


def _player_id_from_linked_cell(value: object) -> str | None:
    if not isinstance(value, tuple) or len(value) < 2:
        return None

    href = value[1]

    if not href:
        return None

    match = re.search(
        r"/players/[^/]+/([^/]+)\.html",
        str(href),
    )

    if match is None:
        return None

    return match.group(1)


def extract_stat_table(
    html: str,
    required_columns: set[str],
) -> pd.DataFrame:
    # extract_links keeps each player's Basketball-Reference ID in
    # the Player cell, avoiding ambiguous name-based matching.
    try:
        tables = pd.read_html(
            StringIO(html),
            displayed_only=False,
            extract_links="body",
        )
    except TypeError:
        # Compatibility fallback for older pandas versions.
        tables = pd.read_html(
            StringIO(html),
            displayed_only=False,
        )

    for frame in tables:
        result = frame.copy()

        result.columns = [
            clean_column_name(column)
            for column in result.columns
        ]

        if not required_columns.issubset(result.columns):
            continue

        player_cells = result["Player"].copy()

        result["player_id"] = player_cells.map(
            _player_id_from_linked_cell
        )

        for column in result.columns:
            if column == "player_id":
                continue

            result[column] = result[column].map(
                _cell_text
            )

        result = result[
            result["Player"].astype(str) != "Player"
        ].copy()

        result = result.dropna(
            subset=["Player"]
        )

        result["Player"] = result["Player"].map(
            normalize_player_name
        )

        return result.reset_index(drop=True)

    available = [
        list(map(clean_column_name, frame.columns))
        for frame in tables
    ]

    raise ValueError(
        "Could not find the expected playoff table. "
        f"Required columns: {sorted(required_columns)}. "
        f"Available tables: {available}"
    )


def normalize_position(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def build_player_lookup() -> pd.DataFrame:
    path = SOURCE_DIR / "Player Totals.csv"

    regular = pd.read_csv(
        path,
        low_memory=False,
    )

    regular["season"] = pd.to_numeric(
        regular["season"],
        errors="coerce",
    ).astype("Int64")

    regular["lg"] = (
        regular["lg"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    regular = regular[
        (regular["lg"] == "NBA")
        & regular["season"].between(
            START_END_YEAR,
            END_END_YEAR,
            inclusive="both",
        )
    ].copy()

    regular["player"] = regular["player"].map(
        normalize_player_name
    )

    regular["player_id"] = (
        regular["player_id"]
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )

    regular["team"] = regular["team"].map(
        normalize_team
    )

    regular["age"] = pd.to_numeric(
        regular.get("age"),
        errors="coerce",
    )

    if "pos" in regular.columns:
        regular["pos"] = regular["pos"].map(
            normalize_position
        )
    else:
        regular["pos"] = ""

    # Aggregate rows such as 2TM cannot identify a playoff team.
    aggregate_team = regular["team"].str.fullmatch(
        r"\d+TM",
        na=False,
    )

    lookup = regular.loc[
        ~aggregate_team,
        [
            "season",
            "player",
            "team",
            "age",
            "pos",
            "player_id",
        ],
    ].dropna(
        subset=[
            "season",
            "player",
            "team",
            "player_id",
        ]
    )

    # Deliberately retain genuine same-name/same-team players. The
    # resolver below uses direct HTML IDs first, then age/position.
    return lookup.drop_duplicates().reset_index(drop=True)


def _fill_unique_player_ids(
    result: pd.DataFrame,
    lookup: pd.DataFrame,
    keys: list[str],
) -> None:
    missing_mask = result["player_id"].isna()

    if not missing_mask.any():
        return

    eligible_mask = missing_mask & result[keys].notna().all(axis=1)

    if not eligible_mask.any():
        return

    candidates = lookup.dropna(
        subset=keys + ["player_id"]
    )

    unique_candidates = (
        candidates.groupby(
            keys,
            dropna=False,
        )["player_id"]
        .agg(
            lambda values: (
                values.iloc[0]
                if values.nunique() == 1
                else pd.NA
            )
        )
        .dropna()
    )

    if unique_candidates.empty:
        return

    lookup_index = pd.MultiIndex.from_frame(
        result.loc[eligible_mask, keys]
    )

    resolved = unique_candidates.reindex(
        lookup_index
    ).to_numpy()

    result.loc[
        eligible_mask,
        "player_id",
    ] = resolved


def attach_player_ids(
    frame: pd.DataFrame,
    end_year: int,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["season"] = end_year
    result["player"] = result["Player"].map(
        normalize_player_name
    )

    if "Tm" not in result.columns:
        raise ValueError(
            "The playoff table does not contain a Tm column, "
            "so players cannot be matched safely."
        )

    result["team"] = result["Tm"].map(
        normalize_team
    )

    result["age"] = pd.to_numeric(
        result["Age"] if "Age" in result.columns else np.nan,
        errors="coerce",
    )

    if "Pos" in result.columns:
        result["pos"] = result["Pos"].map(
            normalize_position
        )
    else:
        result["pos"] = ""

    if "player_id" not in result.columns:
        result["player_id"] = pd.NA

    result["player_id"] = (
        result["player_id"]
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )

    season_lookup = lookup[
        lookup["season"] == end_year
    ].copy()

    # Most rows are resolved directly from their HTML player link.
    # These progressively weaker fallbacks are used only when a link
    # is absent, and only when the key maps to one unique player ID.
    fallback_keys = [
        ["season", "player", "team", "age", "pos"],
        ["season", "player", "team", "age"],
        ["season", "player", "team", "pos"],
        ["season", "player", "team"],
        ["season", "player", "age"],
        ["season", "player"],
    ]

    for keys in fallback_keys:
        _fill_unique_player_ids(
            result,
            season_lookup,
            keys,
        )

    still_missing = result[
        result["player_id"].isna()
    ][
        [
            "season",
            "player",
            "team",
            "age",
            "pos",
        ]
    ].drop_duplicates()

    if not still_missing.empty:
        raise ValueError(
            "Could not safely match playoff players to "
            "Basketball-Reference IDs:\n"
            + still_missing.to_string(index=False)
        )

    return result


def create_metadata(
    source: pd.DataFrame,
    measure_type: str,
    per_mode: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        index=source.index
    )

    output["PLAYER_ID"] = [
        numeric_player_id(
            bref_id,
            player_name,
        )
        for bref_id, player_name in zip(
            source["player_id"],
            source["player"],
            strict=False,
        )
    ]

    output["BREF_PLAYER_ID"] = (
        source["player_id"].astype("string")
    )

    output["PLAYER_NAME"] = (
        source["player"].astype("string")
    )

    if "Tm" in source.columns:
        output["TEAM_ABBREVIATION"] = (
            source["Tm"].astype("string")
        )
    else:
        output["TEAM_ABBREVIATION"] = pd.NA

    output["SEASON"] = (
        source["season"]
        .astype(int)
        .map(season_label)
    )

    output["SEASON_TYPE"] = "Playoffs"
    output["MEASURE_TYPE"] = measure_type
    output["PER_MODE"] = per_mode
    output["DATA_SOURCE"] = (
        "basketball_reference_playoffs_html"
    )

    return output


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


def build_base_totals(
    source: pd.DataFrame,
) -> pd.DataFrame:
    output = create_metadata(
        source,
        measure_type="Base",
        per_mode="Totals",
    )

    mapping = {
        "G": "GP",
        "GS": "GS",
        "MP": "MIN",
        "FG": "FGM",
        "FGA": "FGA",
        "FG%": "FG_PCT",
        "3P": "FG3M",
        "3PA": "FG3A",
        "3P%": "FG3_PCT",
        "2P": "FG2M",
        "2PA": "FG2A",
        "2P%": "FG2_PCT",
        "eFG%": "EFG_PCT",
        "FT": "FTM",
        "FTA": "FTA",
        "FT%": "FT_PCT",
        "ORB": "OREB",
        "DRB": "DREB",
        "TRB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
        "PF": "PF",
        "PTS": "PTS",
    }

    return add_numeric_columns(
        output,
        source,
        mapping,
    )


def build_base_per100(
    source: pd.DataFrame,
) -> pd.DataFrame:
    output = create_metadata(
        source,
        measure_type="Base",
        per_mode="Per100Possessions",
    )

    mapping = {
        "G": "GP",
        "GS": "GS",
        "MP": "MIN",
        "FG": "FGM",
        "FGA": "FGA",
        "FG%": "FG_PCT",
        "3P": "FG3M",
        "3PA": "FG3A",
        "3P%": "FG3_PCT",
        "2P": "FG2M",
        "2PA": "FG2A",
        "2P%": "FG2_PCT",
        "eFG%": "EFG_PCT",
        "FT": "FTM",
        "FTA": "FTA",
        "FT%": "FT_PCT",
        "ORB": "OREB",
        "DRB": "DREB",
        "TRB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
        "PF": "PF",
        "PTS": "PTS",
    }

    return add_numeric_columns(
        output,
        source,
        mapping,
    )


def build_advanced(
    source: pd.DataFrame,
    per100: pd.DataFrame,
) -> pd.DataFrame:
    rating_columns = [
        "season",
        "player_id",
        "ORtg",
        "DRtg",
    ]

    ratings = per100[
        [
            column
            for column in rating_columns
            if column in per100.columns
        ]
    ].copy()

    source = source.merge(
        ratings,
        on=[
            "season",
            "player_id",
        ],
        how="left",
        validate="one_to_one",
    )

    output = create_metadata(
        source,
        measure_type="Advanced",
        per_mode="Totals",
    )

    mapping = {
        "G": "GP",
        "GS": "GS",
        "MP": "MIN",
        "PER": "PER",
        "TS%": "TS_PCT",
        "ORB%": "OREB_PCT",
        "DRB%": "DREB_PCT",
        "TRB%": "REB_PCT",
        "AST%": "AST_PCT",
        "STL%": "STL_PCT",
        "BLK%": "BLK_PCT",
        "TOV%": "TM_TOV_PCT",
        "USG%": "USG_PCT",
        "OWS": "OWS",
        "DWS": "DWS",
        "WS": "WS",
        "WS/48": "WS_PER_48",
        "OBPM": "OBPM",
        "DBPM": "DBPM",
        "BPM": "BPM",
        "VORP": "VORP",
        "ORtg": "OFF_RATING",
        "DRtg": "DEF_RATING",
    }

    output = add_numeric_columns(
        output,
        source,
        mapping,
    )

    if {
        "OFF_RATING",
        "DEF_RATING",
    }.issubset(output.columns):
        output["NET_RATING"] = (
            output["OFF_RATING"]
            - output["DEF_RATING"]
        )

    return output


def load_playoff_pages() -> tuple[
    dict[int, dict[str, pd.DataFrame]],
    pd.DataFrame,
]:
    lookup = build_player_lookup()
    session = requests.Session()

    all_tables: dict[
        int,
        dict[str, pd.DataFrame],
    ] = {}

    for end_year in range(
        START_END_YEAR,
        END_END_YEAR + 1,
    ):
        print(
            f"Loading {season_label(end_year)} playoffs"
        )

        season_tables: dict[str, pd.DataFrame] = {}

        for page_name, config in PAGE_TYPES.items():
            url = (
                "https://www.basketball-reference.com/"
                f"playoffs/NBA_{end_year}_"
                f"{config['url_suffix']}.html"
            )

            cache_path = (
                CACHE_DIR
                / f"NBA_{end_year}_{page_name}.html"
            )

            html = download_html(
                session,
                url,
                cache_path,
            )

            table = extract_stat_table(
                html,
                config["required_columns"],
            )

            table = attach_player_ids(
                table,
                end_year,
                lookup,
            )

            season_tables[page_name] = table

        all_tables[end_year] = season_tables

    return all_tables, lookup


def build_historical_playoff_seasons() -> pd.DataFrame:
    all_tables, _ = load_playoff_pages()

    outputs: list[pd.DataFrame] = []

    for _, tables in all_tables.items():
        totals = tables["totals"]
        per100 = tables["per_poss"]
        advanced = tables["advanced"]

        outputs.append(
            build_base_totals(totals)
        )

        outputs.append(
            build_base_per100(per100)
        )

        outputs.append(
            build_advanced(
                advanced,
                per100,
            )
        )

    combined = pd.concat(
        outputs,
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
            "Historical playoff import contains duplicate "
            "player-season rows:\n"
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
    ]

    print(f"\nWrote: {output_path}")
    print(f"Historical playoff rows: {len(combined):,}")
    print(
        "Playoff seasons:",
        combined["SEASON"].min(),
        "through",
        combined["SEASON"].max(),
    )
    print(
        "Michael Jordan playoff seasons:",
        sorted(jordan["SEASON"].unique()),
    )

    return combined


if __name__ == "__main__":
    build_historical_playoff_seasons()