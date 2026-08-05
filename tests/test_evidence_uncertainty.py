from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goatlab.models.evidence_uncertainty import (
    CATEGORIES,
    build_category_uncertainty,
    build_uncertainty_audit_metadata,
    summarize_expert_diagnostics,
    validate_uncertainty_config,
)


def _config() -> dict:
    return {
        "uncertainty_rules_frozen": True,
        "final_simulation_allowed": False,
        "coverage_penalty": 0.75,
        "confidence_penalty": 1.0,
        "expert_policy": {
            "central_score_weight": 0.0,
            "narrow_intervals_only_when_primary_eligible": True,
            "minimum_primary_rows_per_player_side": 1,
            "minimum_source_families": 3,
            "narrowing_factor": 0.85,
        },
        "categories": {
            category: {
                "base_half_width": 5.0,
                "coverage": 0.9,
                "confidence": 0.8,
            }
            for category in CATEGORIES
        },
    }


def _scores() -> pd.DataFrame:
    rows = []
    for player, base in (
        ("Player A", 90.0),
        ("Player B", 80.0),
    ):
        row = {"PLAYER_NAME": player}
        for index, category in enumerate(CATEGORIES):
            row[category] = base - index
        rows.append(row)
    return pd.DataFrame(rows)


def _expert(primary: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player A",
                "SIDE": "offense",
                "DIMENSION": "creation",
                "SOURCE_FAMILIES": 3,
                "PRIMARY_MODEL_ELIGIBLE": primary,
            }
        ]
    )


def test_valid_config_passes() -> None:
    validate_uncertainty_config(_config())


def test_unfrozen_config_fails() -> None:
    config = _config()
    config["uncertainty_rules_frozen"] = False

    with pytest.raises(
        ValueError,
        match="must be frozen",
    ):
        validate_uncertainty_config(config)


def test_missing_category_rule_fails() -> None:
    config = _config()
    del config["categories"]["peak"]

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_uncertainty_config(config)


def test_expert_central_weight_must_be_zero() -> None:
    config = _config()
    config["expert_policy"][
        "central_score_weight"
    ] = 0.1

    with pytest.raises(
        ValueError,
        match="must be zero",
    ):
        validate_uncertainty_config(config)


def test_intervals_preserve_central_scores() -> None:
    scores = _scores()
    result = build_category_uncertainty(
        scores,
        _config(),
    )

    long_scores = scores.melt(
        id_vars="PLAYER_NAME",
        value_vars=list(CATEGORIES),
        var_name="CATEGORY",
        value_name="EXPECTED",
    )

    merged = result.merge(
        long_scores,
        on=["PLAYER_NAME", "CATEGORY"],
        validate="one_to_one",
    )

    assert np.allclose(
        merged["SCORE"],
        merged["EXPECTED"],
    )
    assert not result[
        "MODEL_SCORE_CHANGED"
    ].any()


def test_intervals_are_bounded_and_ordered() -> None:
    result = build_category_uncertainty(
        _scores(),
        _config(),
    )

    assert result["SCORE_LOW"].between(
        0,
        100,
    ).all()
    assert result["SCORE_HIGH"].between(
        0,
        100,
    ).all()
    assert (
        result["SCORE_LOW"]
        <= result["SCORE"]
    ).all()
    assert (
        result["SCORE"]
        <= result["SCORE_HIGH"]
    ).all()


def test_non_primary_expert_does_not_narrow() -> None:
    baseline = build_category_uncertainty(
        _scores(),
        _config(),
    )
    diagnostic = build_category_uncertainty(
        _scores(),
        _config(),
        expert_consensus=_expert(False),
    )

    base = baseline.query(
        "PLAYER_NAME == 'Player A' "
        "and CATEGORY == 'offense'"
    ).iloc[0]
    compared = diagnostic.query(
        "PLAYER_NAME == 'Player A' "
        "and CATEGORY == 'offense'"
    ).iloc[0]

    assert not compared["EXPERT_USED_TO_NARROW"]
    assert np.isclose(
        compared["INTERVAL_HALF_WIDTH"],
        base["INTERVAL_HALF_WIDTH"],
    )


def test_primary_expert_can_narrow_interval() -> None:
    baseline = build_category_uncertainty(
        _scores(),
        _config(),
    )
    narrowed = build_category_uncertainty(
        _scores(),
        _config(),
        expert_consensus=_expert(True),
    )

    base = baseline.query(
        "PLAYER_NAME == 'Player A' "
        "and CATEGORY == 'offense'"
    ).iloc[0]
    compared = narrowed.query(
        "PLAYER_NAME == 'Player A' "
        "and CATEGORY == 'offense'"
    ).iloc[0]

    assert compared["EXPERT_USED_TO_NARROW"]
    assert (
        compared["INTERVAL_HALF_WIDTH"]
        < base["INTERVAL_HALF_WIDTH"]
    )


def test_defense_reliability_overrides_defaults() -> None:
    defense = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player A",
                "DEFENSE_EVIDENCE_COVERAGE": 0.55,
                "DEFENSE_EVIDENCE_CONFIDENCE": 0.45,
            }
        ]
    )

    result = build_category_uncertainty(
        _scores(),
        _config(),
        defense_evidence=defense,
    )

    row = result.query(
        "PLAYER_NAME == 'Player A' "
        "and CATEGORY == 'defense'"
    ).iloc[0]

    assert np.isclose(row["COVERAGE"], 0.55)
    assert np.isclose(row["CONFIDENCE"], 0.45)


def test_diagnostics_keep_zero_weight() -> None:
    diagnostics = summarize_expert_diagnostics(
        _expert(False),
        central_score_weight=0.0,
    )

    assert diagnostics[
        "PRIMARY_SCORE_WEIGHT"
    ].eq(0.0).all()
    assert not diagnostics[
        "USED_IN_CENTRAL_SCORE"
    ].any()
    assert diagnostics[
        "PRIMARY_ELIGIBLE_ROWS"
    ].sum() == 0


def test_metadata_keeps_final_simulation_blocked() -> None:
    config = _config()
    diagnostics = summarize_expert_diagnostics(
        _expert(False)
    )
    uncertainty = build_category_uncertainty(
        _scores(),
        config,
        expert_consensus=_expert(False),
    )

    metadata = build_uncertainty_audit_metadata(
        uncertainty,
        diagnostics,
        config,
    )

    assert metadata["players"] == 2
    assert metadata["categories"] == 9
    assert metadata["uncertainty_rows"] == 18
    assert metadata[
        "expert_primary_eligible_rows"
    ] == 0
    assert metadata[
        "expert_used_in_central_score"
    ] is False
    assert metadata[
        "expert_used_to_narrow_intervals"
    ] is False
    assert metadata[
        "final_simulation_allowed"
    ] is False
