from __future__ import annotations

import pandas as pd

from goatlab.models.impact_evidence import (
    METRIC_SPECS,
    TARGET_PLAYERS,
    UNAVAILABLE_METRICS,
    build_impact_audit,
    build_impact_coverage,
    build_impact_metric_values,
    build_metric_registry,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Michael Jordan",
                "SEASON": "1996-97",
                "SEASON_TYPE": "Regular Season",
                "NET_RATING_y": 13.7,
                "PIE_y": 0.190,
                "PLUS_MINUS": 820.0,
                "WS_PER_48_y": 0.283,
                "BPM_y": 8.9,
                "VORP_y": 8.6,
                "FAMILY_TEAM_IMPACT": None,
                "COVERAGE_TEAM_IMPACT": 0.0,
            },
            {
                "PLAYER_NAME": "Michael Jordan",
                "SEASON": "1984-85",
                "SEASON_TYPE": "Playoffs",
                "NET_RATING_y": None,
                "PIE_y": None,
                "PLUS_MINUS": None,
                "WS_PER_48_y": None,
                "BPM_y": None,
                "VORP_y": None,
                "FAMILY_TEAM_IMPACT": None,
                "COVERAGE_TEAM_IMPACT": 0.0,
            },
            {
                "PLAYER_NAME": "LeBron James",
                "SEASON": "2012-13",
                "SEASON_TYPE": "Regular Season",
                "NET_RATING_y": 12.3,
                "PIE_y": 0.221,
                "PLUS_MINUS": 720.0,
                "WS_PER_48_y": None,
                "BPM_y": None,
                "VORP_y": None,
                "FAMILY_TEAM_IMPACT": None,
                "COVERAGE_TEAM_IMPACT": 0.0,
            },
            {
                "PLAYER_NAME": "LeBron James",
                "SEASON": "2012-13",
                "SEASON_TYPE": "Playoffs",
                "NET_RATING_y": 7.0,
                "PIE_y": 0.205,
                "PLUS_MINUS": 133.0,
                "WS_PER_48_y": None,
                "BPM_y": None,
                "VORP_y": None,
                "FAMILY_TEAM_IMPACT": None,
                "COVERAGE_TEAM_IMPACT": 0.0,
            },
        ]
    )


def test_metric_values_include_all_local_specs() -> None:
    values = build_impact_metric_values(sample_frame())
    assert set(values["METRIC"]) == {
        spec.name for spec in METRIC_SPECS
    }


def test_metric_values_never_become_primary_eligible() -> None:
    values = build_impact_metric_values(sample_frame())
    assert not values["PRIMARY_MODEL_ELIGIBLE"].any()
    assert values["CENTRAL_SCORE_WEIGHT"].eq(0.0).all()


def test_net_rating_is_not_labeled_on_off() -> None:
    registry = build_metric_registry(
        build_impact_coverage(
            build_impact_metric_values(sample_frame())
        )
    )
    row = registry.loc[registry["METRIC"].eq("net_rating")].iloc[0]
    assert "provenance_mismatch" in row["STATUS"]
    assert row["PRIMARY_MODEL_ELIGIBLE"] is False or not bool(
        row["PRIMARY_MODEL_ELIGIBLE"]
    )


def test_unavailable_metrics_are_not_imputed() -> None:
    registry = build_metric_registry(
        build_impact_coverage(
            build_impact_metric_values(sample_frame())
        )
    )
    unavailable = registry[
        registry["METRIC"].isin(UNAVAILABLE_METRICS)
    ]
    assert len(unavailable) == len(UNAVAILABLE_METRICS)
    assert unavailable["AVAILABILITY"].eq("unavailable").all()
    assert unavailable["CENTRAL_SCORE_WEIGHT"].eq(0.0).all()


def test_coverage_has_one_row_per_player_and_local_metric() -> None:
    coverage = build_impact_coverage(
        build_impact_metric_values(sample_frame())
    )
    assert len(coverage) == len(TARGET_PLAYERS) * len(METRIC_SPECS)
    assert set(coverage["PLAYER_NAME"]) == set(TARGET_PLAYERS)


def test_coverage_rates_are_bounded() -> None:
    coverage = build_impact_coverage(
        build_impact_metric_values(sample_frame())
    )
    assert coverage["COVERAGE_RATE"].between(0.0, 1.0).all()


def test_plus_minus_coverage_asymmetry_is_preserved() -> None:
    coverage = build_impact_coverage(
        build_impact_metric_values(sample_frame())
    )
    rows = coverage[coverage["METRIC"].eq("plus_minus")].set_index(
        "PLAYER_NAME"
    )
    assert rows.loc["Michael Jordan", "OBSERVATIONS"] == 1
    assert rows.loc["LeBron James", "OBSERVATIONS"] == 2


def test_team_impact_empty_family_is_detected() -> None:
    _, _, _, metadata = build_impact_audit(sample_frame())
    assert metadata["team_impact_non_null_rows"] == 0
    assert metadata["team_impact_positive_coverage_rows"] == 0
    assert metadata["team_impact_used_in_model"] is False


def test_audit_does_not_change_central_scores() -> None:
    _, _, registry, metadata = build_impact_audit(sample_frame())
    assert metadata["central_scores_changed"] is False
    assert metadata["central_score_weight_total"] == 0.0
    assert registry["CENTRAL_SCORE_WEIGHT"].eq(0.0).all()


def test_final_simulation_remains_blocked_without_release_blocker() -> None:
    _, _, _, metadata = build_impact_audit(sample_frame())
    assert metadata["release_blockers"] == 0
    assert metadata["final_simulation_allowed"] is False
