from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goatlab.models.final_model import (
    build_production_category_scores,
    run_hierarchy_weight_simulation,
    score_frozen_hierarchy,
    validate_final_model_config,
)

CATEGORIES = (
    "peak",
    "prime",
    "longevity",
    "regular_season",
    "playoffs",
    "winning_context",
    "offense",
    "defense",
    "cultural_impact",
)


def _final_config() -> dict:
    return {
        "production_scale": "bounded_logit_tail",
        "production_scale_categories": [
            "peak",
            "prime",
            "longevity",
            "regular_season",
            "playoffs",
            "offense",
        ],
        "native_scale_categories": [
            "defense",
            "winning_context",
            "cultural_impact",
        ],
        "simulation": {
            "simulations": 250_000,
            "random_seed": 23,
            "within_group_concentration": 100.0,
        },
        "production_weights_frozen": True,
        "production_scale_frozen": True,
        "final_simulation_allowed": True,
    }


def _hierarchy() -> dict:
    return {
        "hierarchy_frozen": True,
        "group_caps_frozen": True,
        "production_weights_frozen": True,
        "final_simulation_allowed": True,
        "groups": [
            {
                "name": "performance_arc",
                "cap": 0.5,
                "categories": [
                    {
                        "name": "peak",
                        "within_group_weight": 0.25,
                    },
                    {
                        "name": "prime",
                        "within_group_weight": 0.20,
                    },
                    {
                        "name": "longevity",
                        "within_group_weight": 0.15,
                    },
                    {
                        "name": "regular_season",
                        "within_group_weight": 0.20,
                    },
                    {
                        "name": "playoffs",
                        "within_group_weight": 0.20,
                    },
                ],
            },
            {
                "name": "basketball_value",
                "cap": 0.4,
                "categories": [
                    {
                        "name": "offense",
                        "within_group_weight": 0.45,
                    },
                    {
                        "name": "defense",
                        "within_group_weight": 0.30,
                    },
                    {
                        "name": "winning_context",
                        "within_group_weight": 0.25,
                    },
                ],
            },
            {
                "name": "broader_legacy",
                "cap": 0.1,
                "categories": [
                    {
                        "name": "cultural_impact",
                        "within_group_weight": 1.0,
                    }
                ],
            },
        ],
    }


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_NAME": "LeBron James",
                "peak": 99.7,
                "prime": 99.4,
                "longevity": 100.0,
                "regular_season": 99.8,
                "playoffs": 99.9,
                "winning_context": 58.4,
                "offense": 99.4,
                "defense": 71.4,
                "cultural_impact": 85.7,
            },
            {
                "PLAYER_NAME": "Michael Jordan",
                "peak": 98.8,
                "prime": 98.7,
                "longevity": 98.9,
                "regular_season": 98.8,
                "playoffs": 99.0,
                "winning_context": 60.2,
                "offense": 98.5,
                "defense": 95.4,
                "cultural_impact": 86.6,
            },
        ]
    )


def test_validate_final_model_config_accepts_freeze() -> None:
    validate_final_model_config(_final_config())


def test_validate_final_model_config_requires_native_defense() -> None:
    config = _final_config()
    config["native_scale_categories"].remove("defense")
    config["production_scale_categories"].append("defense")
    with pytest.raises(ValueError, match="Defense|defense"):
        validate_final_model_config(config)


def test_production_scaling_leaves_native_categories_unchanged() -> None:
    source = _scores()
    result = build_production_category_scores(
        source,
        _final_config(),
    )
    indexed_source = source.set_index("PLAYER_NAME")
    indexed_result = result.set_index("PLAYER_NAME")
    for category in (
        "defense",
        "winning_context",
        "cultural_impact",
    ):
        assert np.allclose(
            indexed_result[category],
            indexed_source[category],
        )


def test_production_scaling_changes_tail_percentiles() -> None:
    source = _scores()
    result = build_production_category_scores(
        source,
        _final_config(),
    ).set_index("PLAYER_NAME")
    assert (
        result.loc["Michael Jordan", "peak"]
        < source.set_index("PLAYER_NAME").loc[
            "Michael Jordan",
            "peak",
        ]
    )


def test_frozen_hierarchy_scores_two_players() -> None:
    production = build_production_category_scores(
        _scores(),
        _final_config(),
    )
    scored = score_frozen_hierarchy(
        production,
        _hierarchy(),
    )
    assert len(scored) == 2
    assert set(scored["RANK"].astype(int)) == {1, 2}


def test_hierarchy_simulation_is_reproducible() -> None:
    production = build_production_category_scores(
        _scores(),
        _final_config(),
    )
    first = run_hierarchy_weight_simulation(
        production,
        _hierarchy(),
        simulations=2_000,
        random_seed=23,
        within_group_concentration=100.0,
    )
    second = run_hierarchy_weight_simulation(
        production,
        _hierarchy(),
        simulations=2_000,
        random_seed=23,
        within_group_concentration=100.0,
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])


def test_hierarchy_simulation_preserves_group_caps() -> None:
    production = build_production_category_scores(
        _scores(),
        _final_config(),
    )
    _, _, groups = run_hierarchy_weight_simulation(
        production,
        _hierarchy(),
        simulations=1_000,
        random_seed=23,
        within_group_concentration=100.0,
    )
    assert np.allclose(
        groups["MIN_REALIZED_GROUP_MASS"],
        groups["FROZEN_GROUP_CAP"],
    )
    assert np.allclose(
        groups["MAX_REALIZED_GROUP_MASS"],
        groups["FROZEN_GROUP_CAP"],
    )


def test_hierarchy_simulation_win_rates_sum_to_one() -> None:
    production = build_production_category_scores(
        _scores(),
        _final_config(),
    )
    summary, _, _ = run_hierarchy_weight_simulation(
        production,
        _hierarchy(),
        simulations=1_000,
        random_seed=23,
        within_group_concentration=100.0,
    )
    assert abs(summary["WIN_RATE"].sum() - 1.0) <= 1e-12
