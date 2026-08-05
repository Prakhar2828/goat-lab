from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.release_gate import (
    evaluate_v1_release_gate,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _hierarchy(path: Path) -> None:
    payload = {
        "hierarchy_frozen": True,
        "group_caps_frozen": True,
        "production_weights_frozen": True,
        "final_simulation_allowed": True,
        "groups": [
            {
                "name": "performance_arc",
                "cap": 0.5,
                "categories": [
                    {"name": "peak", "within_group_weight": 0.25},
                    {"name": "prime", "within_group_weight": 0.20},
                    {"name": "longevity", "within_group_weight": 0.15},
                    {"name": "regular_season", "within_group_weight": 0.20},
                    {"name": "playoffs", "within_group_weight": 0.20},
                ],
            },
            {
                "name": "basketball_value",
                "cap": 0.4,
                "categories": [
                    {"name": "offense", "within_group_weight": 0.45},
                    {"name": "defense", "within_group_weight": 0.30},
                    {"name": "winning_context", "within_group_weight": 0.25},
                ],
            },
            {
                "name": "broader_legacy",
                "cap": 0.1,
                "categories": [
                    {"name": "cultural_impact", "within_group_weight": 1.0}
                ],
            },
        ],
    }
    _write_json(path, payload)


def _final(path: Path) -> None:
    payload = {
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
            "simulations": 250000,
            "random_seed": 23,
            "within_group_concentration": 100.0,
        },
        "mandatory_audits": [
            "mandatory.json",
        ],
        "advisory_audits": [
            "expert.json",
        ],
        "production_weights_frozen": True,
        "production_scale_frozen": True,
        "final_simulation_allowed": True,
    }
    _write_json(path, payload)


def _scores() -> pd.DataFrame:
    rows = []
    for player, offset in (
        ("LeBron James", 1.0),
        ("Michael Jordan", 0.0),
    ):
        rows.append(
            {
                "PLAYER_NAME": player,
                "peak": 98.0 + offset,
                "prime": 97.0 + offset,
                "longevity": 96.0 + offset,
                "regular_season": 95.0 + offset,
                "playoffs": 94.0 + offset,
                "winning_context": 60.0 + offset,
                "offense": 93.0 + offset,
                "defense": 80.0 + offset,
                "cultural_impact": 85.0 + offset,
            }
        )
    return pd.DataFrame(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed = tmp_path / "processed"
    processed.mkdir()

    hierarchy = tmp_path / "hierarchy.json"
    final = tmp_path / "final.json"
    _hierarchy(hierarchy)
    _final(final)

    _write_json(
        processed / "category_scaling_audit.json",
        {
            "scenarios": [
                "historical_percentile",
                "bounded_logit_tail",
            ]
        },
    )
    _write_json(
        processed / "mandatory.json",
        {
            "release_blockers": 0,
            "central_scores_changed": False,
            "additional_central_weight_total": 0.0,
        },
    )
    _write_json(
        processed / "expert.json",
        {
            "release_blockers": 64,
        },
    )
    return processed, hierarchy, final


def test_release_gate_passes_complete_fixture(tmp_path) -> None:
    processed, hierarchy, final = _fixture(tmp_path)
    _, metadata, production = evaluate_v1_release_gate(
        processed_dir=processed,
        category_scores=_scores(),
        hierarchy_config_path=hierarchy,
        final_model_config_path=final,
    )
    assert metadata["release_blockers"] == 0
    assert metadata["final_simulation_allowed"] is True
    assert len(production) == 2


def test_release_gate_blocks_mandatory_audit_failure(tmp_path) -> None:
    processed, hierarchy, final = _fixture(tmp_path)
    _write_json(
        processed / "mandatory.json",
        {"release_blockers": 1},
    )
    _, metadata, _ = evaluate_v1_release_gate(
        processed_dir=processed,
        category_scores=_scores(),
        hierarchy_config_path=hierarchy,
        final_model_config_path=final,
    )
    assert metadata["release_blockers"] > 0
    assert metadata["final_simulation_allowed"] is False


def test_expert_blockers_remain_advisory(tmp_path) -> None:
    processed, hierarchy, final = _fixture(tmp_path)
    checks, metadata, _ = evaluate_v1_release_gate(
        processed_dir=processed,
        category_scores=_scores(),
        hierarchy_config_path=hierarchy,
        final_model_config_path=final,
    )
    expert = checks[
        checks["CHECK"].str.contains(
            "expert.json",
            regex=False,
        )
    ]
    assert not expert["RELEASE_BLOCKER"].any()
    assert metadata["final_simulation_allowed"] is True
