"""Frozen evidence-uncertainty policy for GOAT Lab v1.

The module wraps existing category scores in transparent uncertainty
intervals. It never changes a central category score. Expert film evidence
may narrow an offense or defense interval only when the corresponding
consensus rows are already marked PRIMARY_MODEL_ELIGIBLE.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATEGORIES: tuple[str, ...] = (
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

EXPERT_SIDES: dict[str, str] = {
    "offense": "offense",
    "defense": "defense",
}


def load_uncertainty_config(
    path: str | Path,
) -> dict[str, Any]:
    """Read and validate the frozen uncertainty configuration."""

    config = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    validate_uncertainty_config(config)
    return config


def validate_uncertainty_config(
    config: Mapping[str, Any],
) -> None:
    """Reject incomplete, unfrozen, or numerically invalid policies."""

    if config.get("uncertainty_rules_frozen") is not True:
        raise ValueError(
            "Uncertainty rules must be frozen."
        )

    if config.get("final_simulation_allowed") is not False:
        raise ValueError(
            "Final simulation must remain blocked."
        )

    categories = config.get("categories")
    if not isinstance(categories, Mapping):
        raise ValueError(
            "Configuration is missing category rules."
        )

    missing = set(CATEGORIES).difference(categories)
    extra = set(categories).difference(CATEGORIES)
    if missing or extra:
        raise ValueError(
            "Category rules do not match the model categories: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for category in CATEGORIES:
        rule = categories[category]
        if not isinstance(rule, Mapping):
            raise ValueError(
                f"Category rule must be an object: {category}"
            )

        half_width = float(rule["base_half_width"])
        coverage = float(rule["coverage"])
        confidence = float(rule["confidence"])

        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError(
                f"Invalid base half-width for {category}."
            )

        for label, value in (
            ("coverage", coverage),
            ("confidence", confidence),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Invalid {label} for {category}."
                )

    expert = config.get("expert_policy")
    if not isinstance(expert, Mapping):
        raise ValueError(
            "Configuration is missing expert policy."
        )

    if float(expert.get("central_score_weight", -1)) != 0:
        raise ValueError(
            "Expert evidence central-score weight must be zero."
        )

    factor = float(expert.get("narrowing_factor", np.nan))
    if not np.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError(
            "Expert narrowing factor must be in (0, 1]."
        )


def _numeric_bool(
    values: pd.Series,
) -> pd.Series:
    """Normalize common boolean representations."""

    if values.dtype == bool:
        return values.fillna(False)

    return (
        values.fillna(False)
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes"})
    )


def summarize_expert_diagnostics(
    expert_consensus: pd.DataFrame | None,
    *,
    central_score_weight: float = 0.0,
) -> pd.DataFrame:
    """Summarize film evidence without promoting it into the model."""

    columns = [
        "PLAYER_NAME",
        "SIDE",
        "CONSENSUS_ROWS",
        "DIMENSIONS",
        "SOURCE_FAMILIES",
        "PRIMARY_ELIGIBLE_ROWS",
        "PRIMARY_SCORE_WEIGHT",
        "USED_IN_CENTRAL_SCORE",
    ]

    if expert_consensus is None or expert_consensus.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "PLAYER_NAME",
        "SIDE",
        "DIMENSION",
        "SOURCE_FAMILIES",
        "PRIMARY_MODEL_ELIGIBLE",
    }
    missing = required.difference(expert_consensus.columns)
    if missing:
        raise ValueError(
            "Expert consensus is missing columns: "
            f"{sorted(missing)}"
        )

    frame = expert_consensus.copy()
    frame["_PRIMARY"] = _numeric_bool(
        frame["PRIMARY_MODEL_ELIGIBLE"]
    )
    frame["_SOURCE_FAMILIES"] = pd.to_numeric(
        frame["SOURCE_FAMILIES"],
        errors="coerce",
    ).fillna(0)

    rows: list[dict[str, Any]] = []
    for (player, side), group in frame.groupby(
        ["PLAYER_NAME", "SIDE"],
        dropna=False,
        sort=True,
    ):
        rows.append(
            {
                "PLAYER_NAME": str(player),
                "SIDE": str(side),
                "CONSENSUS_ROWS": len(group),
                "DIMENSIONS": int(
                    group["DIMENSION"].nunique()
                ),
                "SOURCE_FAMILIES": int(
                    group["_SOURCE_FAMILIES"].max()
                ),
                "PRIMARY_ELIGIBLE_ROWS": int(
                    group["_PRIMARY"].sum()
                ),
                "PRIMARY_SCORE_WEIGHT": float(
                    central_score_weight
                ),
                "USED_IN_CENTRAL_SCORE": False,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _eligible_expert_rows(
    expert_consensus: pd.DataFrame | None,
    *,
    player_name: str,
    side: str,
) -> pd.DataFrame:
    if expert_consensus is None or expert_consensus.empty:
        return pd.DataFrame()

    required = {
        "PLAYER_NAME",
        "SIDE",
        "SOURCE_FAMILIES",
        "PRIMARY_MODEL_ELIGIBLE",
    }
    missing = required.difference(expert_consensus.columns)
    if missing:
        raise ValueError(
            "Expert consensus is missing columns: "
            f"{sorted(missing)}"
        )

    selected = expert_consensus[
        expert_consensus["PLAYER_NAME"]
        .astype(str)
        .eq(player_name)
        & expert_consensus["SIDE"]
        .astype(str)
        .str.casefold()
        .eq(side.casefold())
    ].copy()

    if selected.empty:
        return selected

    return selected[
        _numeric_bool(
            selected["PRIMARY_MODEL_ELIGIBLE"]
        )
    ].copy()


def _defense_reliability(
    defense_evidence: pd.DataFrame | None,
    player_name: str,
) -> tuple[float, float] | None:
    if defense_evidence is None or defense_evidence.empty:
        return None

    required = {
        "PLAYER_NAME",
        "DEFENSE_EVIDENCE_COVERAGE",
        "DEFENSE_EVIDENCE_CONFIDENCE",
    }
    if not required.issubset(defense_evidence.columns):
        return None

    selected = defense_evidence[
        defense_evidence["PLAYER_NAME"]
        .astype(str)
        .eq(player_name)
    ]
    if selected.empty:
        return None

    row = selected.iloc[0]
    coverage = float(
        pd.to_numeric(
            row["DEFENSE_EVIDENCE_COVERAGE"],
            errors="coerce",
        )
    )
    confidence = float(
        pd.to_numeric(
            row["DEFENSE_EVIDENCE_CONFIDENCE"],
            errors="coerce",
        )
    )

    if not (
        np.isfinite(coverage)
        and np.isfinite(confidence)
    ):
        return None

    return (
        float(np.clip(coverage, 0, 1)),
        float(np.clip(confidence, 0, 1)),
    )


def build_category_uncertainty(
    category_scores: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    expert_consensus: pd.DataFrame | None = None,
    defense_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build bounded intervals while preserving every central score."""

    validate_uncertainty_config(config)

    required = {"PLAYER_NAME", *CATEGORIES}
    missing = required.difference(category_scores.columns)
    if missing:
        raise ValueError(
            "Category scores are missing columns: "
            f"{sorted(missing)}"
        )

    if category_scores["PLAYER_NAME"].duplicated().any():
        raise ValueError(
            "Category scores contain duplicate players."
        )

    coverage_penalty = float(
        config.get("coverage_penalty", 0.75)
    )
    confidence_penalty = float(
        config.get("confidence_penalty", 1.0)
    )
    expert_policy = config["expert_policy"]

    minimum_rows = int(
        expert_policy[
            "minimum_primary_rows_per_player_side"
        ]
    )
    minimum_families = int(
        expert_policy["minimum_source_families"]
    )
    narrowing_factor = float(
        expert_policy["narrowing_factor"]
    )

    rows: list[dict[str, Any]] = []

    for player_row in category_scores.itertuples(
        index=False
    ):
        player_name = str(player_row.PLAYER_NAME)

        for category in CATEGORIES:
            score = float(
                pd.to_numeric(
                    getattr(player_row, category),
                    errors="coerce",
                )
            )
            if not np.isfinite(score):
                raise ValueError(
                    "Category score must be finite: "
                    f"{player_name} / {category}"
                )
            if not 0 <= score <= 100:
                raise ValueError(
                    "Category score must be in [0, 100]: "
                    f"{player_name} / {category}"
                )

            rule = config["categories"][category]
            coverage = float(rule["coverage"])
            confidence = float(rule["confidence"])

            if category == "defense":
                reliability = _defense_reliability(
                    defense_evidence,
                    player_name,
                )
                if reliability is not None:
                    coverage, confidence = reliability

            half_width = float(
                rule["base_half_width"]
            ) * (
                1
                + coverage_penalty * (1 - coverage)
                + confidence_penalty * (1 - confidence)
            )

            expert_primary_rows = 0
            expert_source_families = 0
            expert_used_to_narrow = False

            side = EXPERT_SIDES.get(category)
            if side is not None:
                eligible = _eligible_expert_rows(
                    expert_consensus,
                    player_name=player_name,
                    side=side,
                )
                expert_primary_rows = len(eligible)

                if not eligible.empty:
                    expert_source_families = int(
                        pd.to_numeric(
                            eligible["SOURCE_FAMILIES"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .max()
                    )

                if (
                    expert_primary_rows >= minimum_rows
                    and expert_source_families
                    >= minimum_families
                ):
                    half_width *= narrowing_factor
                    expert_used_to_narrow = True

            half_width = float(max(half_width, 0))
            score_low = float(
                np.clip(score - half_width, 0, 100)
            )
            score_high = float(
                np.clip(score + half_width, 0, 100)
            )

            rows.append(
                {
                    "PLAYER_NAME": player_name,
                    "CATEGORY": category,
                    "SCORE": score,
                    "SCORE_LOW": score_low,
                    "SCORE_HIGH": score_high,
                    "COVERAGE": coverage,
                    "CONFIDENCE": confidence,
                    "INTERVAL_HALF_WIDTH": half_width,
                    "EXPERT_PRIMARY_ROWS": (
                        expert_primary_rows
                    ),
                    "EXPERT_SOURCE_FAMILIES": (
                        expert_source_families
                    ),
                    "EXPERT_USED_TO_NARROW": (
                        expert_used_to_narrow
                    ),
                    "MODEL_SCORE_CHANGED": False,
                    "EVIDENCE_STATUS": (
                        "expert_primary_narrowing"
                        if expert_used_to_narrow
                        else "base_policy_interval"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["PLAYER_NAME", "CATEGORY"]
    ).reset_index(drop=True)


def build_uncertainty_audit_metadata(
    uncertainty: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create release-gate metadata for the audit runner."""

    primary_rows = int(
        diagnostics["PRIMARY_ELIGIBLE_ROWS"].sum()
    ) if not diagnostics.empty else 0

    expert_narrowing = bool(
        uncertainty["EXPERT_USED_TO_NARROW"].any()
    ) if not uncertainty.empty else False

    return {
        "players": int(
            uncertainty["PLAYER_NAME"].nunique()
        ),
        "categories": int(
            uncertainty["CATEGORY"].nunique()
        ),
        "uncertainty_rows": len(uncertainty),
        "uncertainty_rules_frozen": bool(
            config["uncertainty_rules_frozen"]
        ),
        "expert_primary_eligible_rows": primary_rows,
        "expert_used_in_central_score": False,
        "expert_used_to_narrow_intervals": (
            expert_narrowing
        ),
        "release_blockers": 0,
        "final_simulation_allowed": bool(
            config["final_simulation_allowed"]
        ),
    }
