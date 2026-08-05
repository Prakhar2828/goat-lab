from __future__ import annotations

import json

import pandas as pd
import pytest

from goatlab.models.cultural_impact import (
    calculate_attention_scores,
    calculate_rubric_scores,
    combine_cultural_scores,
)
from goatlab.models.cultural_impact_sensitivity import (
    build_cultural_sensitivity_grid,
    configured_blend_crossover,
    load_policy,
    write_cultural_sensitivity_outputs,
)

DIMENSIONS = (
    "commercial_global_reach",
    "basketball_culture_influence",
    "media_entertainment_reach",
    "philanthropy_social_institutions",
)


def _policy() -> dict:
    return {
        "attention_weight_values": [0.0, 0.2, 0.5, 0.8, 1.0],
        "attention_component_scenarios": {
            "configured": {
                "total_view_share": 0.50,
                "median_daily_view_share": 0.30,
                "median_annual_view_share": 0.20,
            },
            "equal": {
                "total_view_share": 1 / 3,
                "median_daily_view_share": 1 / 3,
                "median_annual_view_share": 1 / 3,
            },
            "total_heavy": {
                "total_view_share": 0.70,
                "median_daily_view_share": 0.20,
                "median_annual_view_share": 0.10,
            },
            "durable_heavy": {
                "total_view_share": 0.20,
                "median_daily_view_share": 0.40,
                "median_annual_view_share": 0.40,
            },
        },
        "rubric_dimension_scenarios": {
            "configured": {
                "commercial_global_reach": 0.30,
                "basketball_culture_influence": 0.30,
                "media_entertainment_reach": 0.15,
                "philanthropy_social_institutions": 0.25,
            },
            "equal": {dimension: 0.25 for dimension in DIMENSIONS},
            "legacy": {
                "commercial_global_reach": 0.30,
                "basketball_culture_influence": 0.45,
                "media_entertainment_reach": 0.15,
                "philanthropy_social_institutions": 0.10,
            },
            "institutions": {
                "commercial_global_reach": 0.20,
                "basketball_culture_influence": 0.20,
                "media_entertainment_reach": 0.20,
                "philanthropy_social_institutions": 0.40,
            },
        },
        "confidence_values": {
            "High": 1.0,
            "Medium": 0.75,
            "Low": 0.5,
        },
        "baseline": {
            "attention_weight": 0.20,
            "attention_component_scenario": "configured",
            "rubric_dimension_scenario": "configured",
        },
    }


def _pageviews() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    for player, values in {
        "Michael Jordan": [100, 110, 105, 95, 100, 110, 105, 95],
        "LeBron James": [110, 115, 112, 108, 110, 115, 112, 108],
    }.items():
        for date, views in zip(dates, values, strict=True):
            rows.append(
                {
                    "PLAYER_NAME": player,
                    "article_requested": player.replace(" ", "_"),
                    "article": player.replace(" ", "_"),
                    "date": date,
                    "views": views,
                }
            )
    return pd.DataFrame(rows)


def _rubric() -> pd.DataFrame:
    values = {
        "Michael Jordan": [98, 100, 94, 90],
        "LeBron James": [90, 94, 98, 98],
    }
    rows = []
    for player, scores in values.items():
        for dimension, score in zip(DIMENSIONS, scores, strict=True):
            rows.append(
                {
                    "PLAYER_NAME": player,
                    "DIMENSION": dimension,
                    "SCORE_0_100": score,
                    "CONFIDENCE": "High",
                    "SOURCE_IDS": "TEST",
                    "RATIONALE": "Synthetic test evidence.",
                }
            )
    return pd.DataFrame(rows)


def _current_scores(policy: dict) -> pd.DataFrame:
    attention = calculate_attention_scores(
        _pageviews(),
        component_weights=policy["attention_component_scenarios"]["configured"],
    )
    rubric = calculate_rubric_scores(
        _rubric(),
        dimension_weights=policy["rubric_dimension_scenarios"]["configured"],
        confidence_values=policy["confidence_values"],
    )
    return combine_cultural_scores(
        attention,
        rubric,
        attention_weight=0.20,
        rubric_weight=0.80,
    )


def test_policy_grid_has_eighty_scenarios() -> None:
    policy = _policy()
    grid, summary, metadata = build_cultural_sensitivity_grid(
        _pageviews(),
        _rubric(),
        _current_scores(policy),
        policy,
    )
    assert len(summary) == 80
    assert len(grid) == 160
    assert metadata["scenarios"] == 80


def test_every_scenario_has_two_players() -> None:
    policy = _policy()
    grid, _, _ = build_cultural_sensitivity_grid(
        _pageviews(),
        _rubric(),
        _current_scores(policy),
        policy,
    )
    assert grid.groupby("SCENARIO_ID").size().eq(2).all()


def test_baseline_reproduces_current_scores() -> None:
    policy = _policy()
    current = _current_scores(policy)
    grid, _, metadata = build_cultural_sensitivity_grid(
        _pageviews(),
        _rubric(),
        current,
        policy,
    )
    baseline = grid.loc[
        grid["IS_BASELINE"],
        ["PLAYER_NAME", "cultural_impact_raw"],
    ].merge(
        current[["PLAYER_NAME", "cultural_impact_raw"]],
        on="PLAYER_NAME",
        suffixes=("_AUDIT", "_CURRENT"),
    )
    error = (
        baseline["cultural_impact_raw_AUDIT"]
        - baseline["cultural_impact_raw_CURRENT"]
    ).abs().max()
    assert error <= 1e-10
    assert metadata["baseline_match_max_abs_error"] <= 1e-10


def test_grid_contains_both_winners() -> None:
    policy = _policy()
    _, summary, metadata = build_cultural_sensitivity_grid(
        _pageviews(),
        _rubric(),
        _current_scores(policy),
        policy,
    )
    assert "Michael Jordan" in set(summary["WINNER"])
    assert "LeBron James" in set(summary["WINNER"])
    assert metadata["winner_robust_across_grid"] is False


def test_scenarios_are_diagnostic_only() -> None:
    policy = _policy()
    grid, summary, metadata = build_cultural_sensitivity_grid(
        _pageviews(),
        _rubric(),
        _current_scores(policy),
        policy,
    )
    assert not grid["PRIMARY_MODEL_ELIGIBLE"].any()
    assert grid["ADDITIONAL_CENTRAL_WEIGHT"].eq(0.0).all()
    assert not summary["PRIMARY_MODEL_ELIGIBLE"].any()
    assert summary["ADDITIONAL_CENTRAL_WEIGHT"].eq(0.0).all()
    assert metadata["central_scores_changed"] is False


def test_configured_crossover_is_inside_unit_interval() -> None:
    policy = _policy()
    attention = calculate_attention_scores(
        _pageviews(),
        component_weights=policy["attention_component_scenarios"]["configured"],
    )
    rubric = calculate_rubric_scores(
        _rubric(),
        dimension_weights=policy["rubric_dimension_scenarios"]["configured"],
        confidence_values=policy["confidence_values"],
    )
    crossover = configured_blend_crossover(attention, rubric)
    assert crossover is not None
    assert 0.0 < crossover < 1.0


def test_outputs_are_written(tmp_path) -> None:
    policy = _policy()
    grid, summary, metadata = build_cultural_sensitivity_grid(
        _pageviews(),
        _rubric(),
        _current_scores(policy),
        policy,
    )
    write_cultural_sensitivity_outputs(
        grid,
        summary,
        metadata,
        tmp_path,
    )
    assert (
        tmp_path / "cultural_impact_sensitivity_grid.parquet"
    ).exists()
    assert (
        tmp_path / "cultural_impact_sensitivity_summary.parquet"
    ).exists()
    assert (
        tmp_path / "cultural_impact_sensitivity_audit.json"
    ).exists()
    loaded = json.loads(
        (
            tmp_path / "cultural_impact_sensitivity_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert loaded["scenarios"] == 80


def test_load_policy_rejects_non_normalized_weights(tmp_path) -> None:
    policy = _policy()
    policy["attention_component_scenarios"]["configured"][
        "total_view_share"
    ] = 0.8
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_policy(path)
