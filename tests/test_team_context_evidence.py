from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goatlab.models.team_context_evidence import (
    attach_context_to_candidate_series,
    build_context_registry,
    build_supporting_cast_context,
    collapse_player_season_rows,
    summarize_supporting_cast,
)


def _season_values() -> pd.DataFrame:
    rows = [
        {
            "PLAYER_ID": 1,
            "PLAYER_NAME": "Michael Jordan",
            "TEAM_ID": 10,
            "SEASON": "1990-91",
            "SEASON_TYPE": "Regular Season",
            "MIN": 3000,
            "SEASON_VALUE_0_100": 95.0,
        },
        {
            "PLAYER_ID": 1,
            "PLAYER_NAME": "Michael Jordan",
            "TEAM_ID": 10,
            "SEASON": "1990-91",
            "SEASON_TYPE": "Regular Season",
            "MIN": 40,
            "SEASON_VALUE_0_100": 10.0,
        },
    ]
    for index, (minutes, value) in enumerate(
        [(2200, 80), (1800, 70), (1500, 60), (1200, 50), (900, 40), (700, 30)],
        start=2,
    ):
        rows.append(
            {
                "PLAYER_ID": index,
                "PLAYER_NAME": f"Teammate {index}",
                "TEAM_ID": 10,
                "SEASON": "1990-91",
                "SEASON_TYPE": "Regular Season",
                "MIN": minutes,
                "SEASON_VALUE_0_100": value,
            }
        )
    return pd.DataFrame(rows)


def test_collapse_player_rows_keeps_largest_minutes() -> None:
    collapsed = collapse_player_season_rows(_season_values())
    jordan = collapsed[collapsed["PLAYER_NAME"].eq("Michael Jordan")]
    assert len(jordan) == 1
    assert jordan.iloc[0]["MIN"] == pytest.approx(3000)


def test_supporting_cast_excludes_focal_player() -> None:
    result = build_supporting_cast_context(_season_values())
    row = result.iloc[0]
    assert row["TEAMMATE_ROWS"] == 6
    assert row["TOP_N_TEAMMATES"] == 6


def test_supporting_cast_weighted_value() -> None:
    result = build_supporting_cast_context(_season_values())
    expected = np.average(
        [80, 70, 60, 50, 40, 30],
        weights=[2200, 1800, 1500, 1200, 900, 700],
    )
    assert result.iloc[0]["SUPPORT_VALUE"] == pytest.approx(expected)


def test_supporting_cast_is_diagnostic_only() -> None:
    result = build_supporting_cast_context(_season_values())
    assert not result["PRIMARY_MODEL_ELIGIBLE"].any()
    assert result["ADDITIONAL_CENTRAL_WEIGHT"].eq(0.0).all()
    assert not result["INJURY_CONTEXT_AVAILABLE"].any()


def test_missing_teammate_values_reduce_coverage_not_score_as_zero() -> None:
    frame = _season_values()
    frame.loc[frame["PLAYER_NAME"].eq("Teammate 2"), "SEASON_VALUE_0_100"] = np.nan
    result = build_supporting_cast_context(frame)
    row = result.iloc[0]
    assert 0.0 < row["SUPPORT_VALUE_COVERAGE"] < 1.0
    assert row["SUPPORT_VALUE"] > 0.0


def test_registry_marks_health_unavailable() -> None:
    registry = build_context_registry()
    health = registry[registry["COMPONENT"].eq("roster_health")].iloc[0]
    assert health["AVAILABILITY"] == "unavailable"
    assert not health["PRIMARY_MODEL_ELIGIBLE"]


def test_registry_adds_no_new_weight() -> None:
    registry = build_context_registry()
    assert registry["ADDITIONAL_CENTRAL_WEIGHT"].sum() == pytest.approx(0.0)


def test_attach_context_does_not_change_expectation() -> None:
    support = build_supporting_cast_context(_season_values())
    series = pd.DataFrame(
        [
            {
                "SERIES_ID": "x",
                "PLAYER_NAME": "Michael Jordan",
                "SEASON": "1990-91",
                "TEAM_ID": 10,
                "EXPECTED_SERIES_WIN_PROB": 0.75,
                "SERIES_OVERPERFORMANCE": 0.25,
            }
        ]
    )
    result = attach_context_to_candidate_series(series, support)
    assert result.iloc[0]["EXPECTED_SERIES_WIN_PROB"] == pytest.approx(0.75)
    assert not result["SUPPORT_CONTEXT_USED_IN_EXPECTATION"].any()
    assert result["ADDITIONAL_CENTRAL_WEIGHT"].eq(0.0).all()


def test_player_summary_reports_zero_injury_rows() -> None:
    support = build_supporting_cast_context(_season_values())
    summary = summarize_supporting_cast(support)
    assert summary.iloc[0]["INJURY_CONTEXT_ROWS"] == 0
