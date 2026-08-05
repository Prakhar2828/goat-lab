from __future__ import annotations

import json

from goatlab.features.build_features import build_player_feature_table
from goatlab.features.team_context import build_team_context
from goatlab.models.category_hierarchy import (
    load_hierarchy_config,
)
from goatlab.models.final_model import (
    load_final_model_config,
    run_hierarchy_weight_simulation,
    score_frozen_hierarchy,
)
from goatlab.models.release_gate import (
    assert_v1_release_gate,
)
from goatlab.reporting.category_scores import build_category_scores
from goatlab.settings import settings
from goatlab.utils import write_parquet


def build_features(include_team_context: bool = True) -> None:
    settings.ensure_directories()
    build_player_feature_table()
    if include_team_context:
        try:
            build_team_context()
        except FileNotFoundError:
            pass
    build_category_scores()


def train_models() -> None:
    settings.ensure_directories()
    category_scores = build_category_scores()

    gate_metadata, production_scores = (
        assert_v1_release_gate(
            processed_dir=settings.processed_dir,
            category_scores=category_scores,
        )
    )

    final_config = load_final_model_config(
        "configs/v1_final_model.json"
    )
    hierarchy = load_hierarchy_config(
        final_config["hierarchy_config"]
    )
    central = score_frozen_hierarchy(
        production_scores,
        hierarchy,
    )

    simulation = final_config["simulation"]
    summary, drivers, group_audit = (
        run_hierarchy_weight_simulation(
            production_scores,
            hierarchy,
            simulations=int(
                simulation["simulations"]
            ),
            random_seed=int(
                simulation["random_seed"]
            ),
            within_group_concentration=float(
                simulation[
                    "within_group_concentration"
                ]
            ),
        )
    )

    write_parquet(
        production_scores,
        settings.processed_dir
        / "production_category_scores.parquet",
    )
    write_parquet(
        central,
        settings.processed_dir
        / "production_hierarchy_scores.parquet",
    )
    write_parquet(
        summary,
        settings.processed_dir
        / "weight_simulation_summary.parquet",
    )
    write_parquet(
        drivers,
        settings.processed_dir
        / "weight_simulation_drivers.parquet",
    )
    write_parquet(
        group_audit,
        settings.processed_dir
        / "hierarchy_weight_simulation_group_audit.parquet",
    )

    metadata = {
        "simulations": int(
            simulation["simulations"]
        ),
        "random_seed": int(
            simulation["random_seed"]
        ),
        "within_group_concentration": float(
            simulation[
                "within_group_concentration"
            ]
        ),
        "production_scale": final_config[
            "production_scale"
        ],
        "production_scale_categories": (
            final_config[
                "production_scale_categories"
            ]
        ),
        "native_scale_categories": (
            final_config[
                "native_scale_categories"
            ]
        ),
        "release_gate": gate_metadata,
        "note": (
            "Playoff expectation training requires "
            "data/manual/playoff_series.csv."
        ),
    }
    (
        settings.model_dir
        / "training_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
