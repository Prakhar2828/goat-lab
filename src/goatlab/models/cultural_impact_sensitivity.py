from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goatlab.models.cultural_impact import (
    calculate_attention_scores,
    calculate_rubric_scores,
    combine_cultural_scores,
)


TARGET_PLAYERS = ("Michael Jordan", "LeBron James")


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen cultural-impact sensitivity policy."""
    policy = json.loads(Path(path).read_text(encoding="utf-8"))

    required = {
        "attention_weight_values",
        "attention_component_scenarios",
        "rubric_dimension_scenarios",
        "confidence_values",
        "baseline",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Cultural sensitivity policy is missing keys: {missing}")

    weights = [float(value) for value in policy["attention_weight_values"]]
    if not weights or any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("Attention weights must be within [0, 1].")

    _validate_scenario_weights(
        policy["attention_component_scenarios"],
        "attention component",
    )
    _validate_scenario_weights(
        policy["rubric_dimension_scenarios"],
        "rubric dimension",
    )
    return policy


def _validate_scenario_weights(
    scenarios: dict[str, dict[str, float]],
    label: str,
) -> None:
    if not scenarios:
        raise ValueError(f"At least one {label} scenario is required.")

    for name, mapping in scenarios.items():
        values = [float(value) for value in mapping.values()]
        if not values:
            raise ValueError(f"{label.title()} scenario {name!r} is empty.")
        if any(value < 0.0 for value in values):
            raise ValueError(
                f"{label.title()} scenario {name!r} has a negative weight."
            )
        if not np.isclose(sum(values), 1.0, atol=1e-10):
            raise ValueError(
                f"{label.title()} scenario {name!r} must sum to 1.0."
            )


def _target_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "PLAYER_NAME" not in frame.columns:
        raise ValueError(f"{label} is missing PLAYER_NAME.")

    target = frame[
        frame["PLAYER_NAME"].isin(TARGET_PLAYERS)
    ].copy()

    observed = set(target["PLAYER_NAME"].dropna().astype(str))
    if observed != set(TARGET_PLAYERS):
        raise ValueError(
            f"{label} must contain exactly the two target players; "
            f"observed={sorted(observed)}"
        )
    return target


def _scenario_id(
    attention_weight: float,
    attention_scenario: str,
    rubric_scenario: str,
) -> str:
    weight_label = f"{attention_weight:.2f}".replace(".", "p")
    return (
        f"attention_{weight_label}"
        f"__{attention_scenario}"
        f"__{rubric_scenario}"
    )


def _winner_from_gap(gap: float, tolerance: float = 1e-12) -> str:
    if abs(gap) <= tolerance:
        return "Tie"
    return "LeBron James" if gap > 0 else "Michael Jordan"


def configured_blend_crossover(
    attention: pd.DataFrame,
    rubric: pd.DataFrame,
) -> float | None:
    """Return the attention weight where the two configured scores tie."""
    attention_target = _target_rows(attention, "attention")
    rubric_target = _target_rows(rubric, "rubric")

    attention_values = attention_target.set_index("PLAYER_NAME")[
        "ATTENTION_SCORE"
    ]
    rubric_values = rubric_target.set_index("PLAYER_NAME")["RUBRIC_SCORE"]

    attention_gap = float(
        attention_values["LeBron James"]
        - attention_values["Michael Jordan"]
    )
    rubric_gap = float(
        rubric_values["LeBron James"]
        - rubric_values["Michael Jordan"]
    )
    denominator = attention_gap - rubric_gap

    if np.isclose(denominator, 0.0, atol=1e-15):
        return None

    crossover = -rubric_gap / denominator
    if crossover < 0.0 or crossover > 1.0:
        return None
    return float(crossover)


def build_cultural_sensitivity_grid(
    pageviews: pd.DataFrame,
    rubric: pd.DataFrame,
    current_scores: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the preregistered cultural-impact weighting grid.

    The function does not alter the production cultural score. Every scenario
    is diagnostic and carries zero additional central-score weight.
    """
    current = _target_rows(current_scores, "current cultural scores")
    current_values = current[
        ["PLAYER_NAME", "cultural_impact_raw"]
    ].copy()

    attention_tables: dict[str, pd.DataFrame] = {}
    for name, component_weights in policy[
        "attention_component_scenarios"
    ].items():
        table = calculate_attention_scores(
            pageviews,
            component_weights=component_weights,
        )
        attention_tables[name] = _target_rows(
            table,
            f"attention scenario {name}",
        )

    rubric_tables: dict[str, pd.DataFrame] = {}
    for name, dimension_weights in policy[
        "rubric_dimension_scenarios"
    ].items():
        table = calculate_rubric_scores(
            rubric,
            dimension_weights=dimension_weights,
            confidence_values=policy["confidence_values"],
        )
        rubric_tables[name] = _target_rows(
            table,
            f"rubric scenario {name}",
        )

    baseline = policy["baseline"]
    baseline_attention_name = str(
        baseline["attention_component_scenario"]
    )
    baseline_rubric_name = str(
        baseline["rubric_dimension_scenario"]
    )
    baseline_attention_weight = float(
        baseline["attention_weight"]
    )

    grid_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for attention_weight_raw in policy["attention_weight_values"]:
        attention_weight = float(attention_weight_raw)
        rubric_weight = 1.0 - attention_weight

        for attention_name, attention_table in attention_tables.items():
            for rubric_name, rubric_table in rubric_tables.items():
                combined = combine_cultural_scores(
                    attention_table,
                    rubric_table,
                    attention_weight=attention_weight,
                    rubric_weight=rubric_weight,
                )
                combined = _target_rows(
                    combined,
                    "combined cultural sensitivity scores",
                )

                scenario_id = _scenario_id(
                    attention_weight,
                    attention_name,
                    rubric_name,
                )
                indexed = combined.set_index("PLAYER_NAME")
                gap = float(
                    indexed.loc[
                        "LeBron James",
                        "cultural_impact_raw",
                    ]
                    - indexed.loc[
                        "Michael Jordan",
                        "cultural_impact_raw",
                    ]
                )
                winner = _winner_from_gap(gap)
                is_baseline = bool(
                    np.isclose(
                        attention_weight,
                        baseline_attention_weight,
                        atol=1e-12,
                    )
                    and attention_name == baseline_attention_name
                    and rubric_name == baseline_rubric_name
                )

                selected_columns = [
                    "PLAYER_NAME",
                    "ATTENTION_SCORE",
                    "RUBRIC_SCORE",
                    "cultural_impact_raw",
                ]
                scenario_rows = combined[selected_columns].copy()
                scenario_rows.insert(0, "SCENARIO_ID", scenario_id)
                scenario_rows["ATTENTION_WEIGHT"] = attention_weight
                scenario_rows["RUBRIC_WEIGHT"] = rubric_weight
                scenario_rows[
                    "ATTENTION_COMPONENT_SCENARIO"
                ] = attention_name
                scenario_rows[
                    "RUBRIC_DIMENSION_SCENARIO"
                ] = rubric_name
                scenario_rows["WINNER"] = winner
                scenario_rows["LEBRON_MINUS_JORDAN"] = gap
                scenario_rows["ABSOLUTE_GAP"] = abs(gap)
                scenario_rows["IS_BASELINE"] = is_baseline
                scenario_rows["PRIMARY_MODEL_ELIGIBLE"] = False
                scenario_rows["ADDITIONAL_CENTRAL_WEIGHT"] = 0.0
                grid_parts.append(scenario_rows)

                summary_rows.append(
                    {
                        "SCENARIO_ID": scenario_id,
                        "ATTENTION_WEIGHT": attention_weight,
                        "RUBRIC_WEIGHT": rubric_weight,
                        "ATTENTION_COMPONENT_SCENARIO": attention_name,
                        "RUBRIC_DIMENSION_SCENARIO": rubric_name,
                        "WINNER": winner,
                        "LEBRON_MINUS_JORDAN": gap,
                        "ABSOLUTE_GAP": abs(gap),
                        "IS_BASELINE": is_baseline,
                        "PRIMARY_MODEL_ELIGIBLE": False,
                        "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
                    }
                )

    grid = pd.concat(grid_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    baseline_rows = grid.loc[
        grid["IS_BASELINE"],
        ["PLAYER_NAME", "cultural_impact_raw"],
    ].merge(
        current_values,
        on="PLAYER_NAME",
        suffixes=("_AUDIT", "_CURRENT"),
        validate="one_to_one",
    )
    baseline_error = float(
        (
            baseline_rows["cultural_impact_raw_AUDIT"]
            - baseline_rows["cultural_impact_raw_CURRENT"]
        )
        .abs()
        .max()
    )

    crossover = configured_blend_crossover(
        attention_tables[baseline_attention_name],
        rubric_tables[baseline_rubric_name],
    )

    winner_counts = {
        str(name): int(count)
        for name, count in summary["WINNER"].value_counts().items()
    }
    blockers: list[str] = []
    if baseline_error > 1e-10:
        blockers.append(
            "baseline_scenario_does_not_reproduce_current_cultural_score"
        )
    if len(summary) != (
        len(policy["attention_weight_values"])
        * len(policy["attention_component_scenarios"])
        * len(policy["rubric_dimension_scenarios"])
    ):
        blockers.append("scenario_grid_size_mismatch")
    if grid["cultural_impact_raw"].isna().any():
        blockers.append("missing_scenario_score")
    if not grid["cultural_impact_raw"].between(0.0, 100.0).all():
        blockers.append("scenario_score_outside_zero_to_one_hundred")

    metadata: dict[str, Any] = {
        "players": 2,
        "scenarios": int(len(summary)),
        "grid_rows": int(len(grid)),
        "attention_weight_values": int(
            len(policy["attention_weight_values"])
        ),
        "attention_component_scenarios": int(
            len(policy["attention_component_scenarios"])
        ),
        "rubric_dimension_scenarios": int(
            len(policy["rubric_dimension_scenarios"])
        ),
        "baseline_match_max_abs_error": baseline_error,
        "configured_blend_crossover_attention_weight": crossover,
        "winner_counts": winner_counts,
        "winner_robust_across_grid": bool(
            summary["WINNER"].nunique() == 1
        ),
        "primary_model_eligible": False,
        "additional_central_weight_total": 0.0,
        "central_scores_changed": False,
        "release_blockers": int(len(blockers)),
        "blocker_details": blockers,
        "final_simulation_allowed": False,
    }
    return grid, summary, metadata


def write_cultural_sensitivity_outputs(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    grid.to_parquet(
        output / "cultural_impact_sensitivity_grid.parquet",
        index=False,
    )
    summary.to_parquet(
        output / "cultural_impact_sensitivity_summary.parquet",
        index=False,
    )
    (
        output / "cultural_impact_sensitivity_audit.json"
    ).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "Cultural-impact sensitivity audit",
        f"Scenarios: {metadata['scenarios']}",
        f"Grid rows: {metadata['grid_rows']}",
        (
            "Configured crossover attention weight: "
            f"{metadata['configured_blend_crossover_attention_weight']}"
        ),
        f"Winner counts: {metadata['winner_counts']}",
        (
            "Winner robust across grid: "
            f"{metadata['winner_robust_across_grid']}"
        ),
        (
            "Baseline maximum absolute error: "
            f"{metadata['baseline_match_max_abs_error']:.12g}"
        ),
        f"Release blockers: {metadata['release_blockers']}",
        "Central scores changed: False",
        "Final simulation allowed: False",
    ]
    (
        output / "cultural_impact_sensitivity_audit.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_cultural_impact_sensitivity(
    *,
    pageviews_path: str | Path = (
        "data/interim/wikimedia_pageviews.parquet"
    ),
    rubric_path: str | Path = "data/manual/cultural_rubric.csv",
    current_scores_path: str | Path = (
        "data/processed/cultural_impact_scores.parquet"
    ),
    policy_path: str | Path = (
        "configs/cultural_impact_sensitivity.json"
    ),
    output_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    policy = load_policy(policy_path)
    pageviews = pd.read_parquet(pageviews_path)
    rubric = pd.read_csv(rubric_path)
    current_scores = pd.read_parquet(current_scores_path)

    grid, summary, metadata = build_cultural_sensitivity_grid(
        pageviews,
        rubric,
        current_scores,
        policy,
    )
    write_cultural_sensitivity_outputs(
        grid,
        summary,
        metadata,
        output_dir,
    )
    return grid, summary, metadata
