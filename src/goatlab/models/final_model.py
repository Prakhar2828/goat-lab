from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goatlab.models.category_hierarchy import (
    CATEGORIES,
    validate_hierarchy_config,
)
from goatlab.models.category_scaling import (
    transform_percentile_scores,
)

TARGET_PLAYERS = ("LeBron James", "Michael Jordan")


def load_final_model_config(
    path: str | Path = "configs/v1_final_model.json",
) -> dict[str, Any]:
    config = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    validate_final_model_config(config)
    return config


def validate_final_model_config(
    config: dict[str, Any],
) -> None:
    if config.get("production_scale") != "bounded_logit_tail":
        raise ValueError(
            "Version 1 production scale must be bounded_logit_tail."
        )

    scaled = tuple(
        str(value)
        for value in config.get(
            "production_scale_categories",
            [],
        )
    )
    native = tuple(
        str(value)
        for value in config.get(
            "native_scale_categories",
            [],
        )
    )

    if set(scaled).intersection(native):
        raise ValueError(
            "Scaled and native category sets must not overlap."
        )

    if set(scaled).union(native) != set(CATEGORIES):
        raise ValueError(
            "Scaled and native categories must cover the complete "
            "production category set."
        )

    if "defense" not in native:
        raise ValueError(
            "Composite defense must remain on its native evidence scale."
        )

    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Final-model simulation configuration is missing.")

    if int(simulation.get("simulations", 0)) != 250_000:
        raise ValueError(
            "Version 1 final simulation count must be 250000."
        )

    if int(simulation.get("random_seed", -1)) != 23:
        raise ValueError(
            "Version 1 final simulation seed must be 23."
        )

    concentration = float(
        simulation.get("within_group_concentration", np.nan)
    )
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError(
            "Within-group Dirichlet concentration must be positive."
        )

    if not bool(config.get("production_weights_frozen", False)):
        raise ValueError("Production weights are not frozen.")

    if not bool(config.get("production_scale_frozen", False)):
        raise ValueError("Production scale is not frozen.")

    if not bool(config.get("final_simulation_allowed", False)):
        raise ValueError("Final simulation is not enabled by the freeze.")


def build_production_category_scores(
    category_scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply the frozen production scale to the current category scores.

    Six categories are historical percentiles and receive the bounded-logit
    tail transform. Defense, winning context, and cultural impact remain on
    their current native 0-100 evidence scales.
    """
    validate_final_model_config(config)

    required = {"PLAYER_NAME", *CATEGORIES}
    missing = required.difference(category_scores.columns)
    if missing:
        raise ValueError(
            "Category scores are missing production inputs: "
            f"{sorted(missing)}"
        )

    result = category_scores[
        ["PLAYER_NAME", *CATEGORIES]
    ].copy()

    for category in config["production_scale_categories"]:
        values = pd.to_numeric(
            result[category],
            errors="coerce",
        )
        result[category] = transform_percentile_scores(
            values,
            str(config["production_scale"]),
        )

    for category in config["native_scale_categories"]:
        result[category] = pd.to_numeric(
            result[category],
            errors="coerce",
        )

    numeric = result[list(CATEGORIES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric.isna().any().any():
        missing_pairs = (
            numeric.isna()
            .stack()
            .loc[lambda values: values]
            .index.tolist()
        )
        raise ValueError(
            "Production category scores contain missing values: "
            f"{missing_pairs}"
        )

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(
            "Production category scores contain non-finite values."
        )

    if not numeric.stack().between(0.0, 100.0).all():
        raise ValueError(
            "Production category scores must remain within [0, 100]."
        )

    return result.sort_values("PLAYER_NAME").reset_index(drop=True)


def frozen_total_weights(
    hierarchy_config: dict[str, Any],
) -> dict[str, float]:
    validate_hierarchy_config(hierarchy_config)
    weights: dict[str, float] = {}

    for group in hierarchy_config["groups"]:
        cap = float(group["cap"])
        for item in group["categories"]:
            category = str(item["name"])
            weights[category] = (
                cap * float(item["within_group_weight"])
            )

    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ValueError("Frozen hierarchy weights must sum to one.")
    return weights


def score_frozen_hierarchy(
    production_scores: pd.DataFrame,
    hierarchy_config: dict[str, Any],
) -> pd.DataFrame:
    weights = frozen_total_weights(hierarchy_config)
    result = production_scores.copy()
    result["GOAT_SCORE"] = sum(
        pd.to_numeric(result[category], errors="coerce")
        * weight
        for category, weight in weights.items()
    )
    result["RANK"] = (
        result["GOAT_SCORE"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    return result.sort_values(
        "GOAT_SCORE",
        ascending=False,
    ).reset_index(drop=True)


def run_hierarchy_weight_simulation(
    production_scores: pd.DataFrame,
    hierarchy_config: dict[str, Any],
    *,
    simulations: int = 250_000,
    random_seed: int = 23,
    within_group_concentration: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sample only within groups while holding group caps exactly fixed."""
    validate_hierarchy_config(hierarchy_config)

    if simulations <= 0:
        raise ValueError("Simulation count must be positive.")
    if within_group_concentration <= 0:
        raise ValueError(
            "Within-group concentration must be positive."
        )

    required = {"PLAYER_NAME", *CATEGORIES}
    missing = required.difference(production_scores.columns)
    if missing:
        raise ValueError(
            "Production scores are missing simulation inputs: "
            f"{sorted(missing)}"
        )

    players = production_scores["PLAYER_NAME"].astype(str).tolist()
    matrix = (
        production_scores
        .set_index("PLAYER_NAME")[list(CATEGORIES)]
        .astype(float)
    )
    if matrix.isna().any().any():
        raise ValueError(
            "Final simulation does not impute missing category scores."
        )

    rng = np.random.default_rng(random_seed)
    category_index = {
        category: index
        for index, category in enumerate(CATEGORIES)
    }
    sampled_weights = np.zeros(
        (simulations, len(CATEGORIES)),
        dtype=float,
    )
    group_rows: list[dict[str, float | str]] = []

    for group in hierarchy_config["groups"]:
        group_name = str(group["name"])
        cap = float(group["cap"])
        items = group["categories"]
        categories = [
            str(item["name"])
            for item in items
        ]
        base = np.asarray(
            [
                float(item["within_group_weight"])
                for item in items
            ],
            dtype=float,
        )

        if len(categories) == 1:
            within_draws = np.ones(
                (simulations, 1),
                dtype=float,
            )
        else:
            alpha = (
                base
                * float(within_group_concentration)
            )
            within_draws = rng.dirichlet(
                alpha,
                size=simulations,
            )

        total_draws = within_draws * cap
        for local_index, category in enumerate(categories):
            sampled_weights[
                :,
                category_index[category],
            ] = total_draws[:, local_index]

        realized_group_mass = total_draws.sum(axis=1)
        group_rows.append(
            {
                "GROUP": group_name,
                "FROZEN_GROUP_CAP": cap,
                "MIN_REALIZED_GROUP_MASS": float(
                    realized_group_mass.min()
                ),
                "MAX_REALIZED_GROUP_MASS": float(
                    realized_group_mass.max()
                ),
                "MEAN_REALIZED_GROUP_MASS": float(
                    realized_group_mass.mean()
                ),
            }
        )

    row_sums = sampled_weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise ValueError(
            "Hierarchy-aware sampled weights do not sum to one."
        )

    score_matrix = (
        sampled_weights
        @ matrix.to_numpy().T
    )
    winner_indices = np.argmax(
        score_matrix,
        axis=1,
    )

    summary = pd.DataFrame(
        {
            "PLAYER_NAME": players,
            "WIN_RATE": [
                float(
                    np.mean(
                        winner_indices == index
                    )
                )
                for index in range(len(players))
            ],
            "MEAN_SCORE": score_matrix.mean(axis=0),
            "P05_SCORE": np.quantile(
                score_matrix,
                0.05,
                axis=0,
            ),
            "P95_SCORE": np.quantile(
                score_matrix,
                0.95,
                axis=0,
            ),
            "SIMULATIONS": int(simulations),
            "RANDOM_SEED": int(random_seed),
            "WITHIN_GROUP_CONCENTRATION": float(
                within_group_concentration
            ),
        }
    )

    category_to_group = {
        str(item["name"]): str(group["name"])
        for group in hierarchy_config["groups"]
        for item in group["categories"]
    }
    frozen = frozen_total_weights(hierarchy_config)

    if len(players) == 2:
        margin = (
            score_matrix[:, 0]
            - score_matrix[:, 1]
        )
        correlations = []
        for index in range(len(CATEGORIES)):
            standard_deviation = float(
                sampled_weights[:, index].std()
            )
            if np.isclose(
                standard_deviation,
                0.0,
                atol=1e-15,
            ):
                correlations.append(np.nan)
            else:
                correlations.append(
                    float(
                        np.corrcoef(
                            sampled_weights[:, index],
                            margin,
                        )[0, 1]
                    )
                )
    else:
        correlations = [np.nan] * len(CATEGORIES)

    drivers = pd.DataFrame(
        {
            "CATEGORY": list(CATEGORIES),
            "GROUP": [
                category_to_group[category]
                for category in CATEGORIES
            ],
            "FROZEN_TOTAL_WEIGHT": [
                frozen[category]
                for category in CATEGORIES
            ],
            "MEAN_SAMPLED_WEIGHT": sampled_weights.mean(
                axis=0
            ),
            "P05_SAMPLED_WEIGHT": np.quantile(
                sampled_weights,
                0.05,
                axis=0,
            ),
            "P95_SAMPLED_WEIGHT": np.quantile(
                sampled_weights,
                0.95,
                axis=0,
            ),
            "MARGIN_CORRELATION_PLAYER_1": correlations,
        }
    ).sort_values(
        "MARGIN_CORRELATION_PLAYER_1",
        key=lambda values: values.abs(),
        ascending=False,
        na_position="last",
    )

    group_audit = pd.DataFrame(group_rows)
    return summary, drivers, group_audit
