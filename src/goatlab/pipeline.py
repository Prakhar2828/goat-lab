from __future__ import annotations

import json

from goatlab.features.build_features import build_player_feature_table
from goatlab.features.team_context import build_team_context
from goatlab.models.sensitivity import run_weight_simulation
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
    if category_scores.drop(columns=["PLAYER_NAME"]).isna().any().any():
        missing = category_scores.set_index("PLAYER_NAME").columns[
            category_scores.set_index("PLAYER_NAME").isna().any()
        ].tolist()
        raise ValueError(
            "Complete the contextual category inputs before final simulation. "
            f"Missing categories: {missing}"
        )
    summary, drivers = run_weight_simulation(category_scores)
    write_parquet(summary, settings.processed_dir / "weight_simulation_summary.parquet")
    write_parquet(drivers, settings.processed_dir / "weight_simulation_drivers.parquet")
    metadata = {
        "simulations": 250000,
        "random_seed": settings.random_seed,
        "note": "Playoff expectation training requires data/manual/playoff_series.csv.",
    }
    (settings.model_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
