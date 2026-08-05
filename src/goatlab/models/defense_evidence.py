from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_PLAYERS = {
    893: "Michael Jordan",
    2544: "LeBron James",
}

AWARD_POINTS = {
    "defensive_player_of_year": 5.0,
    "all_defensive_first": 2.0,
    "all_defensive_second": 1.0,
    "all_defensive_generic": 1.5,
}

AWARD_SCORE_BENCHMARK_POINTS = 25.0
DEFENSE_DIMENSION_COUNT = 16


@dataclass(frozen=True)
class AwardColumns:
    player_id: str | None
    player_name: str | None
    first_name: str | None
    last_name: str | None
    description: str
    season: str | None
    team_number: str | None


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        match = available.get(candidate.casefold())
        if match is not None:
            return match
    return None


def detect_award_columns(awards: pd.DataFrame) -> AwardColumns:
    description = _first_present(
        awards.columns,
        (
            "DESCRIPTION",
            "AWARD_DESCRIPTION",
            "AWARD",
            "TYPE",
        ),
    )
    if description is None:
        raise ValueError(
            "Player awards data has no recognizable award-description column."
        )

    return AwardColumns(
        player_id=_first_present(
            awards.columns,
            (
                "PERSON_ID",
                "PLAYER_ID",
            ),
        ),
        player_name=_first_present(
            awards.columns,
            (
                "PLAYER_NAME",
                "DISPLAY_FIRST_LAST",
            ),
        ),
        first_name=_first_present(
            awards.columns,
            ("FIRST_NAME",),
        ),
        last_name=_first_present(
            awards.columns,
            ("LAST_NAME",),
        ),
        description=description,
        season=_first_present(
            awards.columns,
            (
                "SEASON",
                "SEASON_ID",
            ),
        ),
        team_number=_first_present(
            awards.columns,
            (
                "ALL_NBA_TEAM_NUMBER",
                "TEAM_NUMBER",
            ),
        ),
    )


def _build_player_name(
    awards: pd.DataFrame,
    columns: AwardColumns,
) -> pd.Series:
    if columns.player_name is not None:
        return (
            awards[columns.player_name]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if (
        columns.first_name is not None
        and columns.last_name is not None
    ):
        return (
            awards[columns.first_name]
            .fillna("")
            .astype(str)
            .str.strip()
            + " "
            + awards[columns.last_name]
            .fillna("")
            .astype(str)
            .str.strip()
        ).str.strip()

    if columns.player_id is not None:
        numeric_ids = pd.to_numeric(
            awards[columns.player_id],
            errors="coerce",
        )
        return numeric_ids.map(TARGET_PLAYERS).fillna("")

    raise ValueError(
        "Player awards data has no recognizable player identity columns."
    )


def classify_defensive_award(
    description: object,
    team_number: object = None,
) -> str | None:
    text = (
        ""
        if pd.isna(description)
        else str(description).strip().casefold()
    )
    normalized = (
        text.replace("-", " ")
        .replace("–", " ")
        .replace("—", " ")
    )

    if (
        "defensive player of the year" in normalized
        or "dpoy" in normalized
    ):
        return "defensive_player_of_year"

    if (
        "all defensive" not in normalized
        and "all-defensive" not in text
    ):
        return None

    numeric_team = pd.to_numeric(
        pd.Series([team_number]),
        errors="coerce",
    ).iloc[0]

    if (
        "first team" in normalized
        or numeric_team == 1
    ):
        return "all_defensive_first"

    if (
        "second team" in normalized
        or numeric_team == 2
    ):
        return "all_defensive_second"

    return "all_defensive_generic"


def normalize_defensive_awards(
    awards: pd.DataFrame,
) -> pd.DataFrame:
    columns = detect_award_columns(awards)
    result = awards.copy()

    result["PLAYER_NAME"] = _build_player_name(
        result,
        columns,
    )

    if columns.player_id is not None:
        result["PLAYER_ID"] = pd.to_numeric(
            result[columns.player_id],
            errors="coerce",
        ).astype("Int64")
    else:
        reverse_ids = {
            name: player_id
            for player_id, name in TARGET_PLAYERS.items()
        }
        result["PLAYER_ID"] = (
            result["PLAYER_NAME"]
            .map(reverse_ids)
            .astype("Int64")
        )

    target_names = set(TARGET_PLAYERS.values())
    result = result[
        result["PLAYER_NAME"].isin(target_names)
        | result["PLAYER_ID"].isin(TARGET_PLAYERS)
    ].copy()

    result["AWARD_DESCRIPTION"] = (
        result[columns.description]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if columns.season is None:
        result["SEASON"] = "unknown"
    else:
        result["SEASON"] = (
            result[columns.season]
            .fillna("unknown")
            .astype(str)
            .str.strip()
        )

    team_number = (
        result[columns.team_number]
        if columns.team_number is not None
        else pd.Series(
            np.nan,
            index=result.index,
        )
    )

    result["AWARD_KEY"] = [
        classify_defensive_award(
            description,
            number,
        )
        for description, number in zip(
            result["AWARD_DESCRIPTION"],
            team_number,
            strict=False,
        )
    ]

    result = result[
        result["AWARD_KEY"].notna()
    ].copy()

    result["AWARD_POINTS"] = (
        result["AWARD_KEY"]
        .map(AWARD_POINTS)
        .astype(float)
    )

    result = (
        result.sort_values(
            [
                "PLAYER_NAME",
                "SEASON",
                "AWARD_KEY",
                "AWARD_DESCRIPTION",
            ]
        )
        .drop_duplicates(
            subset=[
                "PLAYER_NAME",
                "SEASON",
                "AWARD_KEY",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return result[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "SEASON",
            "AWARD_KEY",
            "AWARD_DESCRIPTION",
            "AWARD_POINTS",
        ]
    ]


def build_awards_scores(
    normalized_awards: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []

    for player_id, player_name in TARGET_PLAYERS.items():
        selected = normalized_awards[
            normalized_awards["PLAYER_NAME"].eq(
                player_name
            )
        ]

        counts = (
            selected["AWARD_KEY"]
            .value_counts()
            .to_dict()
        )

        total_points = float(
            selected["AWARD_POINTS"].sum()
        )

        score = float(
            np.clip(
                100
                * total_points
                / AWARD_SCORE_BENCHMARK_POINTS,
                0,
                100,
            )
        )

        rows.append(
            {
                "PLAYER_ID": player_id,
                "PLAYER_NAME": player_name,
                "DPOY": int(
                    counts.get(
                        "defensive_player_of_year",
                        0,
                    )
                ),
                "ALL_DEFENSIVE_FIRST": int(
                    counts.get(
                        "all_defensive_first",
                        0,
                    )
                ),
                "ALL_DEFENSIVE_SECOND": int(
                    counts.get(
                        "all_defensive_second",
                        0,
                    )
                ),
                "ALL_DEFENSIVE_GENERIC": int(
                    counts.get(
                        "all_defensive_generic",
                        0,
                    )
                ),
                "DEFENSIVE_AWARD_ROWS": int(
                    len(selected)
                ),
                "DEFENSIVE_AWARD_POINTS": (
                    total_points
                ),
                "defense_awards_score": score,
                "DEFENSE_AWARDS_COVERAGE": (
                    1.0
                    if len(selected) > 0
                    else 0.0
                ),
                "DEFENSE_AWARDS_CONFIDENCE": (
                    0.95
                    if len(selected) > 0
                    else 0.0
                ),
                "AWARDS_USED_IN_MODEL": (
                    len(selected) > 0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_film_diagnostics(
    expert_consensus: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "PLAYER_NAME",
        "SIDE",
        "DIMENSION",
        "CONSENSUS_SCORE",
        "SOURCE_FAMILIES",
        "PRIMARY_MODEL_ELIGIBLE",
    }
    missing = required.difference(
        expert_consensus.columns
    )
    if missing:
        raise ValueError(
            "Expert consensus is missing columns: "
            f"{sorted(missing)}"
        )

    defense = expert_consensus[
        expert_consensus["SIDE"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .eq("defense")
    ].copy()

    defense["PRIMARY_MODEL_ELIGIBLE"] = (
        defense["PRIMARY_MODEL_ELIGIBLE"]
        .fillna(False)
        .astype(bool)
    )

    rows: list[dict[str, float | int | str | bool]] = []

    for player_name in TARGET_PLAYERS.values():
        selected = defense[
            defense["PLAYER_NAME"].eq(
                player_name
            )
        ]
        primary = selected[
            selected["PRIMARY_MODEL_ELIGIBLE"]
        ]

        unique_dimensions = int(
            selected["DIMENSION"].nunique()
        )
        source_families = int(
            pd.to_numeric(
                selected["SOURCE_FAMILIES"],
                errors="coerce",
            )
            .fillna(0)
            .max()
            if not selected.empty
            else 0
        )

        if primary.empty:
            film_score = np.nan
            film_low = np.nan
            film_high = np.nan
        else:
            weights = pd.to_numeric(
                primary.get(
                    "DEFAULT_WEIGHT",
                    1.0,
                ),
                errors="coerce",
            ).fillna(1.0)
            values = pd.to_numeric(
                primary["CONSENSUS_SCORE"],
                errors="coerce",
            )
            valid = values.notna() & weights.gt(0)
            film_score = (
                float(
                    np.average(
                        values[valid],
                        weights=weights[valid],
                    )
                )
                if valid.any()
                else np.nan
            )
            film_low = float(
                pd.to_numeric(
                    primary.get(
                        "CONSENSUS_LOW",
                        np.nan,
                    ),
                    errors="coerce",
                ).mean()
            )
            film_high = float(
                pd.to_numeric(
                    primary.get(
                        "CONSENSUS_HIGH",
                        np.nan,
                    ),
                    errors="coerce",
                ).mean()
            )

        rows.append(
            {
                "PLAYER_NAME": player_name,
                "DEFENSE_FILM_ROWS": int(
                    len(selected)
                ),
                "DEFENSE_FILM_DIMENSIONS": (
                    unique_dimensions
                ),
                "DEFENSE_FILM_SOURCE_FAMILIES": (
                    source_families
                ),
                "DEFENSE_FILM_PRIMARY_ROWS": int(
                    len(primary)
                ),
                "DEFENSE_FILM_COVERAGE": float(
                    min(
                        unique_dimensions
                        / DEFENSE_DIMENSION_COUNT,
                        1.0,
                    )
                ),
                "DEFENSE_FILM_CONFIDENCE": float(
                    min(
                        source_families / 3,
                        1.0,
                    )
                    if len(primary) > 0
                    else 0.0
                ),
                "defense_film_score": film_score,
                "DEFENSE_FILM_LOW": film_low,
                "DEFENSE_FILM_HIGH": film_high,
                "FILM_USED_IN_MODEL": (
                    len(primary) > 0
                    and pd.notna(film_score)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_defense_evidence_scores(
    awards: pd.DataFrame,
    expert_consensus: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_awards = (
        normalize_defensive_awards(
            awards
        )
    )
    awards_scores = build_awards_scores(
        normalized_awards
    )
    film = build_film_diagnostics(
        expert_consensus
    )

    combined = awards_scores.merge(
        film,
        on="PLAYER_NAME",
        how="outer",
        validate="one_to_one",
    )

    combined["DEFENSE_EVIDENCE_COVERAGE"] = (
        0.60
        * pd.to_numeric(
            combined["DEFENSE_AWARDS_COVERAGE"],
            errors="coerce",
        ).fillna(0)
        + 0.40
        * pd.to_numeric(
            combined["DEFENSE_FILM_COVERAGE"],
            errors="coerce",
        ).fillna(0)
    )

    combined["DEFENSE_EVIDENCE_CONFIDENCE"] = (
        0.60
        * pd.to_numeric(
            combined["DEFENSE_AWARDS_CONFIDENCE"],
            errors="coerce",
        ).fillna(0)
        + 0.40
        * pd.to_numeric(
            combined["DEFENSE_FILM_CONFIDENCE"],
            errors="coerce",
        ).fillna(0)
    )

    combined["DEFENSE_EVIDENCE_STATUS"] = np.where(
        combined["FILM_USED_IN_MODEL"],
        "awards_and_primary_film",
        "awards_with_film_diagnostic_only",
    )

    combined["DEFENSE_RELEASE_BLOCKER"] = (
        ~combined["AWARDS_USED_IN_MODEL"]
    )

    return (
        combined.sort_values(
            "PLAYER_NAME"
        ).reset_index(drop=True),
        normalized_awards,
    )
