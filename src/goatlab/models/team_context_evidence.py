from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_PLAYERS = ("Michael Jordan", "LeBron James")


@dataclass(frozen=True)
class ContextComponent:
    component: str
    availability: str
    status: str
    primary_model_eligible: bool
    additional_central_weight: float


CONTEXT_COMPONENTS = (
    ContextComponent(
        "team_srs",
        "available",
        "already_used_in_cross_fit_series_expectation",
        True,
        0.0,
    ),
    ContextComponent(
        "team_net_rating",
        "available",
        "already_used_in_cross_fit_series_expectation",
        True,
        0.0,
    ),
    ContextComponent(
        "supporting_cast_value",
        "partial",
        "diagnostic_only_era_and_roster_coverage",
        False,
        0.0,
    ),
    ContextComponent(
        "roster_health",
        "unavailable",
        "no_reproducible_historical_injury_ledger",
        False,
        0.0,
    ),
    ContextComponent(
        "top_eight_minutes_available",
        "unavailable",
        "no_reproducible_series_level_availability_data",
        False,
        0.0,
    ),
    ContextComponent(
        "coaching_continuity",
        "unavailable",
        "not_ingested_for_v1",
        False,
        0.0,
    ),
    ContextComponent(
        "preseason_expectation",
        "unavailable",
        "not_ingested_for_v1",
        False,
        0.0,
    ),
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _team_key_columns(frame: pd.DataFrame) -> list[str]:
    if "TEAM_ID" in frame.columns and frame["TEAM_ID"].notna().any():
        return ["TEAM_ID"]
    if "TEAM_ABBREVIATION" in frame.columns:
        return ["TEAM_ABBREVIATION"]
    raise ValueError("A team identifier is required.")


def collapse_player_season_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one representative row per player/team/season/type.

    The NBA ingestion can contain totals, per-mode, and advanced rows for the
    same player-season. The totals-style row is consistently the row with the
    largest MIN value, so it is selected without relying on unstable labels.
    """

    required = {"PLAYER_NAME", "SEASON", "SEASON_TYPE", "MIN"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = frame.copy()
    result["MIN"] = _numeric(result, "MIN")
    result["_ROW_ORDER"] = np.arange(len(result))

    player_key = "PLAYER_ID" if "PLAYER_ID" in result.columns else "PLAYER_NAME"
    group_columns = [
        player_key,
        *_team_key_columns(result),
        "SEASON",
        "SEASON_TYPE",
    ]

    result = result.sort_values(
        [*group_columns, "MIN", "_ROW_ORDER"],
        ascending=[True] * len(group_columns) + [False, True],
        na_position="last",
    )
    result = result.drop_duplicates(group_columns, keep="first")
    return result.drop(columns="_ROW_ORDER").reset_index(drop=True)


def _value_column(frame: pd.DataFrame) -> tuple[str, pd.Series]:
    if "SEASON_VALUE_0_100" in frame.columns:
        return "SEASON_VALUE_0_100", _numeric(frame, "SEASON_VALUE_0_100")
    if "SEASON_VALUE_Z" in frame.columns:
        values = 50.0 + 15.0 * _numeric(frame, "SEASON_VALUE_Z")
        return "SEASON_VALUE_Z", values.clip(0.0, 100.0)
    raise ValueError("No season-value column is available.")


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def build_supporting_cast_context(
    season_values: pd.DataFrame,
    target_players: Iterable[str] = TARGET_PLAYERS,
    top_n: int = 8,
    minimum_coverage: float = 0.50,
) -> pd.DataFrame:
    """Build diagnostic supporting-cast estimates excluding the focal player."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    collapsed = collapse_player_season_rows(season_values)
    value_source, values = _value_column(collapsed)
    collapsed = collapsed.copy()
    collapsed["_SUPPORT_VALUE"] = values

    team_keys = _team_key_columns(collapsed)
    targets = collapsed[
        collapsed["PLAYER_NAME"].isin(tuple(target_players))
    ].copy()

    rows: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        mask = (
            collapsed["SEASON"].eq(target["SEASON"])
            & collapsed["SEASON_TYPE"].eq(target["SEASON_TYPE"])
            & collapsed["PLAYER_NAME"].ne(target["PLAYER_NAME"])
        )
        for team_key in team_keys:
            mask &= collapsed[team_key].eq(target[team_key])

        teammates = collapsed.loc[mask].copy()
        teammates["MIN"] = _numeric(teammates, "MIN")
        teammates = teammates[teammates["MIN"].gt(0)].sort_values(
            "MIN", ascending=False
        )
        top = teammates.head(top_n).copy()

        total_teammate_minutes = float(teammates["MIN"].sum())
        top_minutes = float(top["MIN"].sum())
        available_minutes = float(
            top.loc[top["_SUPPORT_VALUE"].notna(), "MIN"].sum()
        )
        coverage = available_minutes / top_minutes if top_minutes > 0 else 0.0
        top_share = (
            top_minutes / total_teammate_minutes
            if total_teammate_minutes > 0
            else 0.0
        )
        support_value = _weighted_average(
            top["_SUPPORT_VALUE"],
            top["MIN"],
        )
        support_available = bool(
            len(top) >= min(5, top_n)
            and coverage >= minimum_coverage
            and np.isfinite(support_value)
        )

        row: dict[str, object] = {
            "PLAYER_NAME": target["PLAYER_NAME"],
            "SEASON": target["SEASON"],
            "SEASON_TYPE": target["SEASON_TYPE"],
            "TEAMMATE_ROWS": int(len(teammates)),
            "TOP_N": int(top_n),
            "TOP_N_TEAMMATES": int(len(top)),
            "TOTAL_TEAMMATE_MINUTES": total_teammate_minutes,
            "TOP_N_MINUTES": top_minutes,
            "TOP_N_MINUTES_SHARE": float(top_share),
            "SUPPORT_VALUE_SOURCE": value_source,
            "SUPPORT_VALUE": support_value,
            "SUPPORT_VALUE_COVERAGE": float(coverage),
            "SUPPORT_VALUE_AVAILABLE": support_available,
            "ROSTER_HEALTH_SCORE": np.nan,
            "INJURY_CONTEXT_AVAILABLE": False,
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
            "CONTEXT_STATUS": (
                "support_diagnostic_health_missing"
                if support_available
                else "support_insufficient_health_missing"
            ),
        }
        for team_key in team_keys:
            row[team_key] = target[team_key]
        rows.append(row)

    return pd.DataFrame(rows)


def build_context_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "COMPONENT": component.component,
                "AVAILABILITY": component.availability,
                "STATUS": component.status,
                "PRIMARY_MODEL_ELIGIBLE": component.primary_model_eligible,
                "ADDITIONAL_CENTRAL_WEIGHT": component.additional_central_weight,
            }
            for component in CONTEXT_COMPONENTS
        ]
    )


def attach_context_to_candidate_series(
    series_scored: pd.DataFrame,
    supporting_cast: pd.DataFrame,
    target_players: Iterable[str] = TARGET_PLAYERS,
) -> pd.DataFrame:
    required = {
        "PLAYER_NAME",
        "SEASON",
        "TEAM_ID",
        "EXPECTED_SERIES_WIN_PROB",
        "SERIES_OVERPERFORMANCE",
    }
    missing = sorted(required.difference(series_scored.columns))
    if missing:
        raise ValueError(f"Missing series columns: {missing}")

    candidates = series_scored[
        series_scored["PLAYER_NAME"].isin(tuple(target_players))
    ].copy()

    regular = supporting_cast[
        supporting_cast["SEASON_TYPE"].eq("Regular Season")
    ].copy()
    merge_keys = ["PLAYER_NAME", "SEASON"]
    if "TEAM_ID" in regular.columns and "TEAM_ID" in candidates.columns:
        merge_keys.append("TEAM_ID")

    context_columns = [
        *merge_keys,
        "SUPPORT_VALUE",
        "SUPPORT_VALUE_COVERAGE",
        "SUPPORT_VALUE_AVAILABLE",
        "ROSTER_HEALTH_SCORE",
        "INJURY_CONTEXT_AVAILABLE",
        "CONTEXT_STATUS",
    ]
    regular = regular[context_columns].drop_duplicates(merge_keys)
    result = candidates.merge(regular, on=merge_keys, how="left", validate="many_to_one")
    result["SUPPORT_CONTEXT_USED_IN_EXPECTATION"] = False
    result["INJURY_CONTEXT_USED_IN_EXPECTATION"] = False
    result["ADDITIONAL_CENTRAL_WEIGHT"] = 0.0
    return result


def summarize_supporting_cast(supporting_cast: pd.DataFrame) -> pd.DataFrame:
    if supporting_cast.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for player_name, group in supporting_cast.groupby("PLAYER_NAME", sort=True):
        values = _numeric(group, "SUPPORT_VALUE")
        coverage = _numeric(group, "SUPPORT_VALUE_COVERAGE")
        rows.append(
            {
                "PLAYER_NAME": player_name,
                "PLAYER_SEASONS": int(len(group)),
                "REGULAR_SEASONS": int(group["SEASON_TYPE"].eq("Regular Season").sum()),
                "PLAYOFF_SEASONS": int(group["SEASON_TYPE"].eq("Playoffs").sum()),
                "SUPPORT_VALUE_ROWS": int(values.notna().sum()),
                "MEDIAN_SUPPORT_VALUE": float(values.median()) if values.notna().any() else np.nan,
                "MEAN_SUPPORT_COVERAGE": float(coverage.mean()) if coverage.notna().any() else 0.0,
                "INJURY_CONTEXT_ROWS": int(group["INJURY_CONTEXT_AVAILABLE"].sum()),
                "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
            }
        )
    return pd.DataFrame(rows)
