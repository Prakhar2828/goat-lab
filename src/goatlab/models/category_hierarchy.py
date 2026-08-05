from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def load_hierarchy_config(
    path: str | Path = "configs/category_hierarchy.json",
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_hierarchy_config(config)
    return config


def validate_hierarchy_config(config: dict[str, Any]) -> None:
    groups = config.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Category hierarchy must define at least one group.")

    seen: list[str] = []
    caps: list[float] = []

    for group in groups:
        name = str(group.get("name", "")).strip()
        if not name:
            raise ValueError("Every category group requires a name.")

        cap = float(group.get("cap", np.nan))
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError(f"Group {name!r} has an invalid cap.")
        caps.append(cap)

        categories = group.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"Group {name!r} has no categories.")

        within: list[float] = []
        for item in categories:
            category = str(item.get("name", "")).strip()
            weight = float(item.get("within_group_weight", np.nan))

            if not category:
                raise ValueError(f"Group {name!r} contains an unnamed category.")
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Category {category!r} has an invalid within-group weight."
                )

            seen.append(category)
            within.append(weight)

        if not np.isclose(sum(within), 1.0, atol=1e-9):
            raise ValueError(f"Within-group weights for {name!r} must sum to 1.")

    if len(seen) != len(set(seen)):
        raise ValueError("Every category must appear in exactly one group.")

    if set(seen) != set(CATEGORIES):
        missing = sorted(set(CATEGORIES).difference(seen))
        extra = sorted(set(seen).difference(CATEGORIES))
        raise ValueError(
            "Category hierarchy does not match the production category set. "
            f"Missing={missing}; extra={extra}."
        )

    if not np.isclose(sum(caps), 1.0, atol=1e-9):
        raise ValueError("Category group caps must sum to 1.")

    if bool(config.get("final_simulation_allowed", False)):
        raise ValueError("This patch may not unlock the final simulation.")


def build_weight_table(config: dict[str, Any]) -> pd.DataFrame:
    validate_hierarchy_config(config)

    rows: list[dict[str, float | str | bool]] = []
    for group in config["groups"]:
        group_cap = float(group["cap"])
        for item in group["categories"]:
            within = float(item["within_group_weight"])
            rows.append(
                {
                    "GROUP": str(group["name"]),
                    "GROUP_DISPLAY_NAME": str(
                        group.get("display_name", group["name"])
                    ),
                    "CATEGORY": str(item["name"]),
                    "GROUP_CAP": group_cap,
                    "WITHIN_GROUP_WEIGHT": within,
                    "PROVISIONAL_TOTAL_WEIGHT": group_cap * within,
                    "HIERARCHY_FROZEN": bool(config["hierarchy_frozen"]),
                    "PRODUCTION_WEIGHT_FROZEN": bool(
                        config["production_weights_frozen"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def score_category_hierarchy(
    category_scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    validate_hierarchy_config(config)

    required = {"PLAYER_NAME", *CATEGORIES}
    missing = required.difference(category_scores.columns)
    if missing:
        raise ValueError(
            "Category scores are missing hierarchy inputs: "
            f"{sorted(missing)}"
        )

    rows: list[dict[str, float | str]] = []

    for source_row in category_scores.itertuples(index=False):
        output: dict[str, float | str] = {
            "PLAYER_NAME": str(source_row.PLAYER_NAME)
        }
        weighted_total = 0.0
        available_mass = 0.0
        coverage = 0.0

        for group in config["groups"]:
            group_name = str(group["name"])
            group_cap = float(group["cap"])

            values: list[float] = []
            weights: list[float] = []
            available_within = 0.0

            for item in group["categories"]:
                category = str(item["name"])
                weight = float(item["within_group_weight"])
                value = pd.to_numeric(
                    pd.Series([getattr(source_row, category)]),
                    errors="coerce",
                ).iloc[0]

                if pd.notna(value):
                    values.append(float(value))
                    weights.append(weight)
                    available_within += weight

            output[f"{group_name.upper()}_COVERAGE"] = available_within

            if not values:
                output[f"{group_name.upper()}_SCORE"] = np.nan
                continue

            group_score = float(np.average(values, weights=weights))
            output[f"{group_name.upper()}_SCORE"] = group_score

            mass = group_cap * available_within
            weighted_total += group_score * mass
            available_mass += mass
            coverage += mass

        output["HIERARCHICAL_SCORE"] = (
            weighted_total / available_mass if available_mass > 0 else np.nan
        )
        output["HIERARCHICAL_COVERAGE"] = coverage
        rows.append(output)

    return pd.DataFrame(rows)


def _declared_overlap_map(
    config: dict[str, Any],
) -> dict[frozenset[str], str]:
    declarations: dict[frozenset[str], str] = {}

    for item in config.get("overlap_declarations", []):
        categories = item.get("categories", [])
        if len(categories) != 2:
            raise ValueError(
                "Every overlap declaration must contain exactly two categories."
            )

        key = frozenset(str(value) for value in categories)
        if len(key) != 2:
            raise ValueError(
                "An overlap declaration cannot repeat the same category."
            )

        declarations[key] = str(item.get("rationale", "")).strip()

    return declarations


def build_overlap_audit(
    historical_reference: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    validate_hierarchy_config(config)

    reference_columns = config.get("reference_columns", {})
    if not isinstance(reference_columns, dict):
        raise ValueError("reference_columns must be a mapping.")

    missing_columns = set(reference_columns.values()).difference(
        historical_reference.columns
    )
    if missing_columns:
        raise ValueError(
            "Historical reference is missing overlap columns: "
            f"{sorted(missing_columns)}"
        )

    threshold = float(config.get("correlation_advisory_threshold", 0.80))
    minimum_rows = int(config.get("minimum_reference_rows", 30))
    declarations = _declared_overlap_map(config)

    rows: list[dict[str, float | int | str | bool]] = []

    for category_a, category_b in combinations(reference_columns, 2):
        column_a = str(reference_columns[category_a])
        column_b = str(reference_columns[category_b])

        pair = historical_reference[[column_a, column_b]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()

        sample_size = int(len(pair))
        correlation = (
            float(pair[column_a].corr(pair[column_b], method="spearman"))
            if sample_size >= 2
            else np.nan
        )
        absolute = abs(correlation) if np.isfinite(correlation) else np.nan

        key = frozenset([category_a, category_b])
        declared = key in declarations
        blocker = sample_size < minimum_rows or not np.isfinite(correlation)

        if sample_size < minimum_rows:
            status = "insufficient_reference"
        elif not np.isfinite(correlation):
            status = "correlation_unavailable"
        elif declared:
            status = "declared_dependency"
        elif absolute >= threshold:
            status = "high_overlap_advisory"
        else:
            status = "acceptable"

        rows.append(
            {
                "CATEGORY_A": category_a,
                "CATEGORY_B": category_b,
                "REFERENCE_COLUMN_A": column_a,
                "REFERENCE_COLUMN_B": column_b,
                "SAMPLE_SIZE": sample_size,
                "SPEARMAN_CORRELATION": correlation,
                "ABS_SPEARMAN_CORRELATION": absolute,
                "ADVISORY_THRESHOLD": threshold,
                "DECLARED_DEPENDENCY": declared,
                "RATIONALE": declarations.get(key, ""),
                "STATUS": status,
                "RELEASE_BLOCKER": blocker,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "ABS_SPEARMAN_CORRELATION",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )
