from __future__ import annotations

import re

import numpy as np
import pandas as pd


ROUND_NAMES = {
    1: "First Round",
    2: "Conference Semifinals",
    3: "Conference Finals",
    4: "NBA Finals",
}

EXPECTED_SERIES_PER_ROUND = {
    1: 8,
    2: 4,
    3: 2,
    4: 1,
}


def normalize_round_number(
    value: object,
) -> int | None:
    if value is None or pd.isna(value):
        return None

    label = str(value).strip()

    if not label:
        return None

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        label.casefold(),
    ).strip()

    if (
        "nba finals" in normalized
        or (
            normalized == "finals"
        )
    ):
        return 4

    if (
        "conference finals" in normalized
        or "conf finals" in normalized
        or "conference final" in normalized
        or "conf final" in normalized
    ):
        return 3

    if (
        "semifinals" in normalized
        or "semifinal" in normalized
    ):
        return 2

    if "first round" in normalized:
        return 1

    return None


def _first_nonblank(
    values: pd.Series,
) -> str:
    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


def _build_series_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "SERIES_ID",
        "SEASON",
        "TEAM_ID",
        "OPP_TEAM_ID",
        "TEAM_WON_SERIES",
        "SERIES_START_DATE",
        "SERIES_END_DATE",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Playoff series data is missing columns: "
            f"{sorted(missing)}"
        )

    working = frame.copy()

    if "ROUND" not in working.columns:
        working["ROUND"] = ""

    working[
        "TEAM_WON_SERIES"
    ] = pd.to_numeric(
        working["TEAM_WON_SERIES"],
        errors="coerce",
    )

    working["TEAM_ID"] = pd.to_numeric(
        working["TEAM_ID"],
        errors="coerce",
    )

    working[
        "OPP_TEAM_ID"
    ] = pd.to_numeric(
        working["OPP_TEAM_ID"],
        errors="coerce",
    )

    working[
        "SERIES_START_DATE"
    ] = pd.to_datetime(
        working["SERIES_START_DATE"],
        errors="coerce",
    )

    working[
        "SERIES_END_DATE"
    ] = pd.to_datetime(
        working["SERIES_END_DATE"],
        errors="coerce",
    )

    winner_rows = working[
        working[
            "TEAM_WON_SERIES"
        ].eq(1)
    ].copy()

    winner_counts = (
        winner_rows.groupby(
            "SERIES_ID"
        )
        .size()
    )

    invalid_winners = winner_counts[
        winner_counts.ne(1)
    ]

    all_series = set(
        working["SERIES_ID"]
        .astype(str)
    )

    winner_series = set(
        winner_counts.index.astype(str)
    )

    if (
        invalid_winners.any()
        or all_series != winner_series
    ):
        raise ValueError(
            "Every series must have exactly "
            "one winning perspective."
        )

    winners = winner_rows[
        [
            "SERIES_ID",
            "SEASON",
            "TEAM_ID",
            "OPP_TEAM_ID",
        ]
    ].rename(
        columns={
            "TEAM_ID": "WINNER_TEAM_ID",
            "OPP_TEAM_ID": (
                "LOSER_TEAM_ID"
            ),
        }
    )

    metadata = (
        working.groupby(
            "SERIES_ID",
            as_index=False,
        )
        .agg(
            SEASON=(
                "SEASON",
                "first",
            ),
            SERIES_START_DATE=(
                "SERIES_START_DATE",
                "min",
            ),
            SERIES_END_DATE=(
                "SERIES_END_DATE",
                "max",
            ),
            ROUND_RAW=(
                "ROUND",
                _first_nonblank,
            ),
        )
    )

    series = metadata.merge(
        winners,
        on=[
            "SERIES_ID",
            "SEASON",
        ],
        how="inner",
        validate="one_to_one",
    )

    if series[
        [
            "SERIES_START_DATE",
            "SERIES_END_DATE",
            "WINNER_TEAM_ID",
            "LOSER_TEAM_ID",
        ]
    ].isna().any().any():
        raise ValueError(
            "Series progression requires dates "
            "and both participating teams."
        )

    series[
        "SOURCE_ROUND_NUMBER"
    ] = series["ROUND_RAW"].map(
        normalize_round_number
    )

    return series


def _infer_season_rounds(
    season_series: pd.DataFrame,
) -> pd.Series:
    season_series = (
        season_series.sort_values(
            [
                "SERIES_START_DATE",
                "SERIES_ID",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    parent: dict[
        str,
        str | None,
    ] = {}

    for row in season_series.itertuples(
        index=False
    ):
        winner = int(
            row.WINNER_TEAM_ID
        )

        candidates = season_series[
            (
                season_series[
                    "SERIES_START_DATE"
                ]
                > row.SERIES_END_DATE
            )
            & (
                season_series[
                    "WINNER_TEAM_ID"
                ].eq(winner)
                | season_series[
                    "LOSER_TEAM_ID"
                ].eq(winner)
            )
        ].sort_values(
            [
                "SERIES_START_DATE",
                "SERIES_ID",
            ]
        )

        parent[str(row.SERIES_ID)] = (
            None
            if candidates.empty
            else str(
                candidates.iloc[0][
                    "SERIES_ID"
                ]
            )
        )

    cache: dict[str, int] = {}

    def resolve(
        series_id: str,
        trail: set[str] | None = None,
    ) -> int:
        if series_id in cache:
            return cache[series_id]

        trail = set(
            trail or set()
        )

        if series_id in trail:
            raise ValueError(
                "Cycle detected in playoff progression."
            )

        trail.add(series_id)

        next_series = parent[
            series_id
        ]

        if next_series is None:
            number = 4
        else:
            number = (
                resolve(
                    next_series,
                    trail,
                )
                - 1
            )

        if number not in ROUND_NAMES:
            raise ValueError(
                "Unable to map series progression "
                f"to a four-round bracket: {series_id}"
            )

        cache[series_id] = number
        return number

    inferred = {
        series_id: resolve(
            series_id
        )
        for series_id in parent
    }

    return pd.Series(
        inferred,
        name="INFERRED_ROUND_NUMBER",
        dtype="int64",
    )


def add_canonical_playoff_rounds(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize source labels and infer missing rounds by progression."""

    series = _build_series_table(
        frame
    )

    inferred_parts: list[
        pd.DataFrame
    ] = []

    for _, group in series.groupby(
        "SEASON",
        sort=True,
    ):
        group = group.copy()

        inferred = _infer_season_rounds(
            group
        )

        group[
            "INFERRED_ROUND_NUMBER"
        ] = (
            group["SERIES_ID"]
            .astype(str)
            .map(inferred)
        )

        inferred_parts.append(
            group
        )

    series = pd.concat(
        inferred_parts,
        ignore_index=True,
    )

    conflict = (
        series[
            "SOURCE_ROUND_NUMBER"
        ].notna()
        & series[
            "SOURCE_ROUND_NUMBER"
        ].astype("Int64")
        .ne(
            series[
                "INFERRED_ROUND_NUMBER"
            ].astype("Int64")
        )
    )

    if conflict.any():
        raise ValueError(
            "Source round labels conflict with "
            "bracket progression:\n"
            + series.loc[
                conflict,
                [
                    "SERIES_ID",
                    "SEASON",
                    "ROUND_RAW",
                    "SOURCE_ROUND_NUMBER",
                    "INFERRED_ROUND_NUMBER",
                ],
            ].to_string(
                index=False
            )
        )

    series["ROUND_NUMBER"] = (
        series[
            "SOURCE_ROUND_NUMBER"
        ]
        .fillna(
            series[
                "INFERRED_ROUND_NUMBER"
            ]
        )
        .astype("int64")
    )

    series["ROUND_SOURCE"] = np.where(
        series[
            "SOURCE_ROUND_NUMBER"
        ].notna(),
        "source_label",
        "inferred_progression",
    )

    series["ROUND"] = (
        series["ROUND_NUMBER"]
        .map(ROUND_NAMES)
    )

    counts = (
        series.groupby(
            [
                "SEASON",
                "ROUND_NUMBER",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=[
                1,
                2,
                3,
                4,
            ],
            fill_value=0,
        )
    )

    expected = pd.Series(
        EXPECTED_SERIES_PER_ROUND
    )

    invalid_seasons = counts[
        counts.ne(
            expected,
            axis="columns",
        ).any(
            axis=1
        )
    ]

    if not invalid_seasons.empty:
        raise ValueError(
            "Invalid playoff-round counts:\n"
            + invalid_seasons.to_string()
        )

    metadata = series[
        [
            "SERIES_ID",
            "ROUND_RAW",
            "ROUND",
            "ROUND_NUMBER",
            "ROUND_SOURCE",
        ]
    ]

    result = frame.copy()

    if "ROUND_RAW" not in result.columns:
        if "ROUND" in result.columns:
            result[
                "ROUND_RAW"
            ] = result["ROUND"]
        else:
            result[
                "ROUND_RAW"
            ] = ""

    result = result.drop(
        columns=[
            column
            for column in [
                "ROUND",
                "ROUND_NUMBER",
                "ROUND_SOURCE",
            ]
            if column in result.columns
        ]
    )

    result = result.merge(
        metadata.drop(
            columns=[
                "ROUND_RAW",
            ]
        ),
        on="SERIES_ID",
        how="left",
        validate="many_to_one",
    )

    if result[
        "ROUND_NUMBER"
    ].isna().any():
        raise ValueError(
            "Some playoff rows still have "
            "no canonical round."
        )

    return result
