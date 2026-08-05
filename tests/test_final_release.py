from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from goatlab.reporting.final_release import (
    build_scale_sensitivity,
    hierarchy_weights,
    render_results_markdown,
    sha256_file,
    weighted_scores,
)


def _hierarchy() -> dict:
    return {
        "groups": [
            {
                "name": "performance",
                "cap": 0.6,
                "categories": [
                    {
                        "name": "peak",
                        "within_group_weight": 1.0,
                    }
                ],
            },
            {
                "name": "value",
                "cap": 0.4,
                "categories": [
                    {
                        "name": "defense",
                        "within_group_weight": 1.0,
                    }
                ],
            },
        ]
    }


def test_hierarchy_weights_sum_to_one() -> None:
    weights = hierarchy_weights(_hierarchy())
    assert weights == {
        "peak": pytest.approx(0.6),
        "defense": pytest.approx(0.4),
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weighted_scores_use_frozen_weights() -> None:
    frame = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "LeBron James",
                "peak": 90.0,
                "defense": 70.0,
            },
            {
                "PLAYER_NAME": "Michael Jordan",
                "peak": 80.0,
                "defense": 100.0,
            },
        ]
    )
    scores = weighted_scores(
        frame,
        hierarchy_weights(_hierarchy()),
    )
    assert scores["LeBron James"] == pytest.approx(82.0)
    assert scores["Michael Jordan"] == pytest.approx(88.0)


def test_scale_sensitivity_keeps_native_categories_native() -> None:
    comparison = pd.DataFrame(
        [
            {
                "SCENARIO": "a",
                "PLAYER_NAME": "LeBron James",
                "CATEGORY": "peak",
                "SCORE": 100.0,
            },
            {
                "SCENARIO": "a",
                "PLAYER_NAME": "Michael Jordan",
                "CATEGORY": "peak",
                "SCORE": 90.0,
            },
            {
                "SCENARIO": "b",
                "PLAYER_NAME": "LeBron James",
                "CATEGORY": "peak",
                "SCORE": 80.0,
            },
            {
                "SCENARIO": "b",
                "PLAYER_NAME": "Michael Jordan",
                "CATEGORY": "peak",
                "SCORE": 95.0,
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "LeBron James",
                "defense": 70.0,
            },
            {
                "PLAYER_NAME": "Michael Jordan",
                "defense": 100.0,
            },
        ]
    )
    config = {
        "production_scale": "a",
        "production_scale_categories": ["peak"],
        "native_scale_categories": ["defense"],
    }

    rows = build_scale_sensitivity(
        comparison,
        current,
        config,
        _hierarchy(),
    )
    assert [row["winner"] for row in rows] == [
        "Michael Jordan",
        "Michael Jordan",
    ]
    assert rows[0]["is_production"] is True
    assert rows[1]["is_production"] is False


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"frozen-goat-lab-result"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_render_markdown_uses_conditional_language() -> None:
    manifest = {
        "source_commit": "abc123",
        "source_branch": "feature/model-integrity",
        "central_result": {
            "winner": "LeBron James",
            "margin_points": 0.1,
            "players": [
                {
                    "player": "LeBron James",
                    "goat_score": 89.2,
                    "rank": 1.0,
                },
                {
                    "player": "Michael Jordan",
                    "goat_score": 89.1,
                    "rank": 2.0,
                },
            ],
        },
        "simulation_result": {
            "winner": "LeBron James",
            "players": [
                {
                    "player": "LeBron James",
                    "win_rate": 0.60,
                    "mean_score": 89.2,
                    "p05_score": 88.2,
                    "p95_score": 90.2,
                },
                {
                    "player": "Michael Jordan",
                    "win_rate": 0.40,
                    "mean_score": 89.1,
                    "p05_score": 88.1,
                    "p95_score": 90.0,
                },
            ],
        },
        "scale_sensitivity": {
            "scenarios": [
                {
                    "scenario": "bounded",
                    "is_production": True,
                    "lebron_score": 89.2,
                    "jordan_score": 89.1,
                    "lebron_minus_jordan": 0.1,
                    "winner": "LeBron James",
                }
            ]
        },
        "model_freeze": {
            "simulations": 250000,
            "random_seed": 23,
            "within_group_concentration": 100.0,
            "production_scale": "bounded",
            "release_gate_passed_checks": 32,
            "release_gate_checks": 32,
        },
        "simulation_drivers": [
            {
                "CATEGORY": "defense",
                "FROZEN_TOTAL_WEIGHT": 0.12,
                "MARGIN_CORRELATION_PLAYER_1": -0.9,
            }
        ],
        "limitations": ["Scaling matters."],
    }
    markdown = render_results_markdown(manifest)
    assert "conditional model result" in markdown
    assert "not an objective probability" in markdown
    assert "Scaling matters." in markdown


def test_required_config_leaf_reads_nested_freeze_values() -> None:
    from goatlab.reporting.final_release import (
        _required_config_leaf,
    )

    config = {
        "simulation": {
            "simulations": 250000,
            "random_seed": 23,
            "within_group_concentration": 100.0,
        },
        "release_gate": {
            "simulations": 250000,
            "random_seed": 23,
        },
    }

    assert (
        _required_config_leaf(
            config,
            "simulations",
        )
        == 250000
    )
    assert (
        _required_config_leaf(
            config,
            "random_seed",
        )
        == 23
    )
    assert (
        _required_config_leaf(
            config,
            "within_group_concentration",
        )
        == 100.0
    )
