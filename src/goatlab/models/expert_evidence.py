from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_PLAYERS = (
    "Michael Jordan",
    "LeBron James",
)

SOURCE_SCORE_LIMITS = {
    "EXPERTISE_SCORE": 3,
    "FILM_SPECIFICITY_SCORE": 3,
    "METHODOLOGY_SCORE": 3,
    "SAMPLE_DISCLOSURE_SCORE": 2,
    "STATISTICAL_SUPPORT_SCORE": 2,
    "CAREER_COVERAGE_SCORE": 2,
    "BALANCED_EVIDENCE_SCORE": 2,
    "INDEPENDENCE_SCORE": 2,
}

SOURCE_REQUIRED_COLUMNS = {
    "SOURCE_ID",
    "SOURCE_FAMILY",
    "ANALYST",
    "PUBLICATION",
    "TITLE",
    "PUBLICATION_DATE",
    "URL",
    "SOURCE_TYPE",
    *SOURCE_SCORE_LIMITS,
}

CLAIM_REQUIRED_COLUMNS = {
    "CLAIM_ID",
    "SOURCE_ID",
    "PLAYER_NAME",
    "CAREER_PHASE",
    "SEASON_START",
    "SEASON_END",
    "SEASON_TYPE",
    "SIDE",
    "DIMENSION",
    "CLAIM_DIRECTION",
    "CLAIM_STRENGTH",
    "EVIDENCE_TYPE",
    "FILM_EXAMPLES_PRESENT",
    "SAMPLE_SIZE_DISCLOSED",
    "CONFIDENCE",
    "SUPPORTING_LOCATION",
    "SUMMARY",
    "LIMITATIONS",
    "REVIEW_STATUS",
}

DIMENSION_REQUIRED_COLUMNS = {
    "SIDE",
    "DIMENSION",
    "DESCRIPTION",
    "MIN_SOURCE_FAMILIES",
    "MIN_TIER_A_FAMILIES",
    "PRIMARY_ELIGIBLE",
    "DEFAULT_WEIGHT",
}

VALID_SIDES = {
    "offense",
    "defense",
}

VALID_DIRECTIONS = {
    "major_strength",
    "strength",
    "mixed",
    "limitation",
    "major_limitation",
}

VALID_REVIEW_STATUSES = {
    "pending",
    "verified",
    "verified_with_qualification",
    "rejected",
}

ACCEPTED_REVIEW_STATUSES = {
    "verified",
    "verified_with_qualification",
}

DIRECTION_BASE_SCORES = {
    "major_limitation": 10.0,
    "limitation": 30.0,
    "mixed": 50.0,
    "strength": 70.0,
    "major_strength": 90.0,
}

CONSENSUS_COLUMNS = [
    "PLAYER_NAME",
    "CAREER_PHASE",
    "SIDE",
    "DIMENSION",
    "CONSENSUS_SCORE",
    "CONSENSUS_LOW",
    "CONSENSUS_HIGH",
    "FAMILY_DISAGREEMENT",
    "SOURCE_FAMILIES",
    "TIER_A_FAMILIES",
    "CLAIMS",
    "PLAYER_COVERAGE_COUNT",
    "MIN_SOURCE_FAMILIES",
    "MIN_TIER_A_FAMILIES",
    "DIMENSION_PRIMARY_ELIGIBLE",
    "DEFAULT_WEIGHT",
    "EVIDENCE_STATUS",
    "PRIMARY_MODEL_ELIGIBLE",
]


def read_expert_evidence(
    manual_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "sources": manual_dir / "expert_sources.csv",
        "claims": manual_dir / "expert_claims.csv",
        "dimensions": manual_dir / "expert_analysis_dimensions.csv",
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing expert-evidence files: "
            f"{missing}"
        )

    return (
        pd.read_csv(paths["sources"]),
        pd.read_csv(paths["claims"]),
        pd.read_csv(paths["dimensions"]),
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{label} is missing columns: "
            f"{sorted(missing)}"
        )


def _coerce_bool(
    values: pd.Series,
    label: str,
) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)

    normalized = (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    invalid = normalized[
        ~normalized.isin(
            mapping
        )
    ]

    if not invalid.empty:
        raise ValueError(
            f"{label} contains invalid booleans: "
            f"{sorted(invalid.unique())}"
        )

    return normalized.map(
        mapping
    ).astype(bool)


def validate_expert_evidence(
    sources: pd.DataFrame,
    claims: pd.DataFrame,
    dimensions: pd.DataFrame,
) -> None:
    _require_columns(
        sources,
        SOURCE_REQUIRED_COLUMNS,
        "expert_sources",
    )

    _require_columns(
        claims,
        CLAIM_REQUIRED_COLUMNS,
        "expert_claims",
    )

    _require_columns(
        dimensions,
        DIMENSION_REQUIRED_COLUMNS,
        "expert_analysis_dimensions",
    )

    source_text_columns = [
        "SOURCE_ID",
        "SOURCE_FAMILY",
        "ANALYST",
        "PUBLICATION",
        "TITLE",
        "URL",
        "SOURCE_TYPE",
    ]

    for column in source_text_columns:
        normalized = (
            sources[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if normalized.eq("").any():
            raise ValueError(
                "expert_sources contains blank "
                f"{column} values."
            )

    urls = (
        sources["URL"]
        .astype(str)
        .str.strip()
    )

    invalid_urls = urls[
        ~urls.str.startswith(
            "https://"
        )
    ]

    if not invalid_urls.empty:
        raise ValueError(
            "Expert source URLs must use HTTPS: "
            f"{sorted(invalid_urls.unique())}"
        )

    publication_dates = pd.to_datetime(
        sources["PUBLICATION_DATE"],
        errors="coerce",
    )

    if publication_dates.isna().any():
        invalid_dates = (
            sources.loc[
                publication_dates.isna(),
                "PUBLICATION_DATE",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Invalid expert-source publication dates: "
            f"{invalid_dates}"
        )

    for frame, identifier, label in [
        (
            sources,
            "SOURCE_ID",
            "expert_sources",
        ),
        (
            claims,
            "CLAIM_ID",
            "expert_claims",
        ),
    ]:
        identifiers = (
            frame[identifier]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if identifiers.eq("").any():
            raise ValueError(
                f"{label} contains blank "
                f"{identifier} values."
            )

        duplicates = identifiers[
            identifiers.duplicated(
                keep=False
            )
        ]

        if not duplicates.empty:
            raise ValueError(
                f"{label} contains duplicate "
                f"{identifier} values: "
                f"{sorted(duplicates.unique())}"
            )

    dimension_keys = dimensions[
        [
            "SIDE",
            "DIMENSION",
        ]
    ].astype(str)

    duplicate_dimensions = dimension_keys[
        dimension_keys.duplicated(
            keep=False
        )
    ]

    if not duplicate_dimensions.empty:
        raise ValueError(
            "expert_analysis_dimensions contains "
            "duplicate SIDE/DIMENSION pairs."
        )

    invalid_sides = set(
        dimensions["SIDE"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    ).difference(
        VALID_SIDES
    )

    if invalid_sides:
        raise ValueError(
            "Invalid dimension sides: "
            f"{sorted(invalid_sides)}"
        )

    for column, maximum in (
        SOURCE_SCORE_LIMITS.items()
    ):
        numeric = pd.to_numeric(
            sources[column],
            errors="coerce",
        )

        if (
            numeric.isna().any()
            or numeric.lt(0).any()
            or numeric.gt(maximum).any()
        ):
            raise ValueError(
                f"{column} must contain values "
                f"between 0 and {maximum}."
            )

    source_ids = set(
        sources["SOURCE_ID"]
        .astype(str)
        .str.strip()
    )

    claim_source_ids = set(
        claims["SOURCE_ID"]
        .astype(str)
        .str.strip()
    )

    unknown_sources = (
        claim_source_ids
        - source_ids
    )

    if unknown_sources:
        raise ValueError(
            "Claims reference unknown sources: "
            f"{sorted(unknown_sources)}"
        )

    claim_sides = set(
        claims["SIDE"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    invalid_claim_sides = (
        claim_sides
        - VALID_SIDES
    )

    if invalid_claim_sides:
        raise ValueError(
            "Invalid claim sides: "
            f"{sorted(invalid_claim_sides)}"
        )

    directions = set(
        claims["CLAIM_DIRECTION"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    invalid_directions = (
        directions
        - VALID_DIRECTIONS
    )

    if invalid_directions:
        raise ValueError(
            "Invalid claim directions: "
            f"{sorted(invalid_directions)}"
        )

    statuses = set(
        claims["REVIEW_STATUS"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    invalid_statuses = (
        statuses
        - VALID_REVIEW_STATUSES
    )

    if invalid_statuses:
        raise ValueError(
            "Invalid review statuses: "
            f"{sorted(invalid_statuses)}"
        )

    strength = pd.to_numeric(
        claims["CLAIM_STRENGTH"],
        errors="coerce",
    )

    if (
        not claims.empty
        and (
            strength.isna().any()
            or strength.lt(1).any()
            or strength.gt(3).any()
        )
    ):
        raise ValueError(
            "CLAIM_STRENGTH must contain "
            "values between 1 and 3."
        )

    confidence = pd.to_numeric(
        claims["CONFIDENCE"],
        errors="coerce",
    )

    if (
        not claims.empty
        and (
            confidence.isna().any()
            or confidence.lt(0).any()
            or confidence.gt(1).any()
        )
    ):
        raise ValueError(
            "CONFIDENCE must contain values "
            "between 0 and 1."
        )

    for boolean_column in [
        "FILM_EXAMPLES_PRESENT",
        "SAMPLE_SIZE_DISCLOSED",
    ]:
        if not claims.empty:
            _coerce_bool(
                claims[boolean_column],
                boolean_column,
            )

    if not claims.empty:
        player_names = set(
            claims["PLAYER_NAME"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        invalid_players = (
            player_names
            - set(TARGET_PLAYERS)
        )

        if invalid_players:
            raise ValueError(
                "Expert claims contain unsupported "
                "players: "
                f"{sorted(invalid_players)}"
            )

        normalized_status = (
            claims["REVIEW_STATUS"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )

        accepted_mask = (
            normalized_status.isin(
                ACCEPTED_REVIEW_STATUSES
            )
        )

        accepted_requirements = {
            "CAREER_PHASE": 3,
            "SEASON_TYPE": 2,
            "EVIDENCE_TYPE": 3,
            "SUPPORTING_LOCATION": 12,
            "SUMMARY": 20,
            "LIMITATIONS": 20,
        }

        for column, minimum_length in (
            accepted_requirements.items()
        ):
            values = (
                claims.loc[
                    accepted_mask,
                    column,
                ]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            invalid = values[
                values.str.len().lt(
                    minimum_length
                )
            ]

            if not invalid.empty:
                raise ValueError(
                    "Accepted expert claims require "
                    f"a substantive {column} value."
                )

    available_dimensions = set(
        zip(
            dimensions["SIDE"]
            .astype(str)
            .str.strip()
            .str.casefold(),
            dimensions["DIMENSION"]
            .astype(str)
            .str.strip(),
            strict=False,
        )
    )

    claim_dimensions = set(
        zip(
            claims["SIDE"]
            .astype(str)
            .str.strip()
            .str.casefold(),
            claims["DIMENSION"]
            .astype(str)
            .str.strip(),
            strict=False,
        )
    )

    unknown_dimensions = (
        claim_dimensions
        - available_dimensions
    )

    if unknown_dimensions:
        raise ValueError(
            "Claims reference unknown dimensions: "
            f"{sorted(unknown_dimensions)}"
        )

    _coerce_bool(
        dimensions["PRIMARY_ELIGIBLE"],
        "PRIMARY_ELIGIBLE",
    )

    for column in [
        "MIN_SOURCE_FAMILIES",
        "MIN_TIER_A_FAMILIES",
    ]:
        numeric = pd.to_numeric(
            dimensions[column],
            errors="coerce",
        )

        if (
            numeric.isna().any()
            or numeric.lt(0).any()
        ):
            raise ValueError(
                f"{column} must contain "
                "nonnegative integers."
            )

    default_weight = pd.to_numeric(
        dimensions["DEFAULT_WEIGHT"],
        errors="coerce",
    )

    if (
        default_weight.isna().any()
        or default_weight.lt(0).any()
    ):
        raise ValueError(
            "DEFAULT_WEIGHT must contain "
            "nonnegative values."
        )


def score_expert_sources(
    sources: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        sources,
        SOURCE_REQUIRED_COLUMNS,
        "expert_sources",
    )

    result = sources.copy()

    for column in SOURCE_SCORE_LIMITS:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    result[
        "SOURCE_QUALITY_SCORE"
    ] = result[
        list(
            SOURCE_SCORE_LIMITS
        )
    ].sum(
        axis=1
    )

    score = result[
        "SOURCE_QUALITY_SCORE"
    ]

    result["SOURCE_TIER"] = np.select(
        [
            score.ge(14),
            score.ge(10),
            score.ge(6),
        ],
        [
            "A",
            "B",
            "C",
        ],
        default="excluded",
    )

    result[
        "SOURCE_QUALITY_WEIGHT"
    ] = (
        score
        / float(
            sum(
                SOURCE_SCORE_LIMITS.values()
            )
        )
    ).clip(
        0,
        1,
    )

    return result


def _weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        numeric_values.notna()
        & numeric_weights.notna()
        & numeric_weights.gt(0)
    )

    if not valid.any():
        return float("nan")

    return float(
        np.average(
            numeric_values.loc[valid],
            weights=numeric_weights.loc[valid],
        )
    )


def _weighted_quantile(
    values: Iterable[float],
    weights: Iterable[float],
    quantile: float,
) -> float:
    value_array = np.asarray(
        list(values),
        dtype=float,
    )

    weight_array = np.asarray(
        list(weights),
        dtype=float,
    )

    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )

    value_array = value_array[
        valid
    ]

    weight_array = weight_array[
        valid
    ]

    if value_array.size == 0:
        return float("nan")

    order = np.argsort(
        value_array
    )

    value_array = value_array[
        order
    ]

    weight_array = weight_array[
        order
    ]

    cumulative = np.cumsum(
        weight_array
    )

    cutoff = (
        float(quantile)
        * cumulative[-1]
    )

    index = int(
        np.searchsorted(
            cumulative,
            cutoff,
            side="left",
        )
    )

    return float(
        value_array[
            min(
                index,
                len(value_array) - 1,
            )
        ]
    )


def _empty_consensus() -> pd.DataFrame:
    return pd.DataFrame(
        columns=CONSENSUS_COLUMNS
    )


def build_expert_consensus(
    sources: pd.DataFrame,
    claims: pd.DataFrame,
    dimensions: pd.DataFrame,
    target_players: tuple[str, ...] = (
        TARGET_PLAYERS
    ),
) -> pd.DataFrame:
    validate_expert_evidence(
        sources,
        claims,
        dimensions,
    )

    if claims.empty:
        return _empty_consensus()

    source_quality = (
        score_expert_sources(
            sources
        )
    )

    merged = claims.merge(
        source_quality[
            [
                "SOURCE_ID",
                "SOURCE_FAMILY",
                "SOURCE_TIER",
                "SOURCE_QUALITY_SCORE",
                "SOURCE_QUALITY_WEIGHT",
            ]
        ],
        on="SOURCE_ID",
        how="left",
        validate="many_to_one",
    )

    merged["REVIEW_STATUS"] = (
        merged["REVIEW_STATUS"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    merged["CLAIM_DIRECTION"] = (
        merged["CLAIM_DIRECTION"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    merged["SIDE"] = (
        merged["SIDE"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    merged = merged[
        merged["REVIEW_STATUS"].isin(
            ACCEPTED_REVIEW_STATUSES
        )
        & merged["SOURCE_TIER"].ne(
            "excluded"
        )
    ].copy()

    if merged.empty:
        return _empty_consensus()

    merged["CLAIM_STRENGTH"] = (
        pd.to_numeric(
            merged["CLAIM_STRENGTH"],
            errors="raise",
        )
    )

    merged["CONFIDENCE"] = (
        pd.to_numeric(
            merged["CONFIDENCE"],
            errors="raise",
        )
    )

    base_score = merged[
        "CLAIM_DIRECTION"
    ].map(
        DIRECTION_BASE_SCORES
    )

    merged["CLAIM_EVIDENCE_SCORE"] = (
        50
        + (
            base_score
            - 50
        )
        * (
            merged[
                "CLAIM_STRENGTH"
            ]
            / 3
        )
        * merged[
            "CONFIDENCE"
        ]
    ).clip(
        0,
        100,
    )

    group_columns = [
        "PLAYER_NAME",
        "CAREER_PHASE",
        "SIDE",
        "DIMENSION",
        "SOURCE_FAMILY",
    ]

    family_rows: list[
        dict[str, float | int | str | bool]
    ] = []

    for keys, group in merged.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            player_name,
            career_phase,
            side,
            dimension,
            source_family,
        ) = keys

        family_rows.append(
            {
                "PLAYER_NAME": str(
                    player_name
                ),
                "CAREER_PHASE": str(
                    career_phase
                ),
                "SIDE": str(
                    side
                ),
                "DIMENSION": str(
                    dimension
                ),
                "SOURCE_FAMILY": str(
                    source_family
                ),
                "FAMILY_SCORE": _weighted_mean(
                    group[
                        "CLAIM_EVIDENCE_SCORE"
                    ],
                    group[
                        "SOURCE_QUALITY_WEIGHT"
                    ],
                ),
                "FAMILY_WEIGHT": float(
                    group[
                        "SOURCE_QUALITY_WEIGHT"
                    ].max()
                ),
                "TIER_A_FAMILY": bool(
                    group[
                        "SOURCE_TIER"
                    ].eq(
                        "A"
                    ).any()
                ),
                "CLAIMS": int(
                    len(group)
                ),
            }
        )

    family_frame = pd.DataFrame(
        family_rows
    )

    consensus_rows: list[
        dict[str, float | int | str]
    ] = []

    consensus_group_columns = [
        "PLAYER_NAME",
        "CAREER_PHASE",
        "SIDE",
        "DIMENSION",
    ]

    for keys, group in family_frame.groupby(
        consensus_group_columns,
        dropna=False,
        sort=True,
    ):
        (
            player_name,
            career_phase,
            side,
            dimension,
        ) = keys

        consensus_score = _weighted_mean(
            group["FAMILY_SCORE"],
            group["FAMILY_WEIGHT"],
        )

        low = _weighted_quantile(
            group["FAMILY_SCORE"],
            group["FAMILY_WEIGHT"],
            0.20,
        )

        high = _weighted_quantile(
            group["FAMILY_SCORE"],
            group["FAMILY_WEIGHT"],
            0.80,
        )

        if len(group) > 1:
            disagreement = float(
                np.sqrt(
                    np.average(
                        (
                            group[
                                "FAMILY_SCORE"
                            ]
                            - consensus_score
                        )
                        ** 2,
                        weights=group[
                            "FAMILY_WEIGHT"
                        ],
                    )
                )
            )
        else:
            disagreement = 0.0

        consensus_rows.append(
            {
                "PLAYER_NAME": str(
                    player_name
                ),
                "CAREER_PHASE": str(
                    career_phase
                ),
                "SIDE": str(
                    side
                ),
                "DIMENSION": str(
                    dimension
                ),
                "CONSENSUS_SCORE": float(
                    consensus_score
                ),
                "CONSENSUS_LOW": float(
                    low
                ),
                "CONSENSUS_HIGH": float(
                    high
                ),
                "FAMILY_DISAGREEMENT": (
                    disagreement
                ),
                "SOURCE_FAMILIES": int(
                    group[
                        "SOURCE_FAMILY"
                    ].nunique()
                ),
                "TIER_A_FAMILIES": int(
                    group[
                        "TIER_A_FAMILY"
                    ].sum()
                ),
                "CLAIMS": int(
                    group[
                        "CLAIMS"
                    ].sum()
                ),
            }
        )

    consensus = pd.DataFrame(
        consensus_rows
    )

    dimension_config = (
        dimensions.copy()
    )

    dimension_config["SIDE"] = (
        dimension_config["SIDE"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    dimension_config[
        "MIN_SOURCE_FAMILIES"
    ] = pd.to_numeric(
        dimension_config[
            "MIN_SOURCE_FAMILIES"
        ],
        errors="raise",
    ).astype(int)

    dimension_config[
        "MIN_TIER_A_FAMILIES"
    ] = pd.to_numeric(
        dimension_config[
            "MIN_TIER_A_FAMILIES"
        ],
        errors="raise",
    ).astype(int)

    dimension_config[
        "PRIMARY_ELIGIBLE"
    ] = _coerce_bool(
        dimension_config[
            "PRIMARY_ELIGIBLE"
        ],
        "PRIMARY_ELIGIBLE",
    )

    dimension_config[
        "DEFAULT_WEIGHT"
    ] = pd.to_numeric(
        dimension_config[
            "DEFAULT_WEIGHT"
        ],
        errors="raise",
    )

    dimension_config = (
        dimension_config.rename(
            columns={
                "PRIMARY_ELIGIBLE": (
                    "DIMENSION_PRIMARY_ELIGIBLE"
                )
            }
        )
    )

    consensus = consensus.merge(
        dimension_config[
            [
                "SIDE",
                "DIMENSION",
                "MIN_SOURCE_FAMILIES",
                "MIN_TIER_A_FAMILIES",
                "DIMENSION_PRIMARY_ELIGIBLE",
                "DEFAULT_WEIGHT",
            ]
        ],
        on=[
            "SIDE",
            "DIMENSION",
        ],
        how="left",
        validate="many_to_one",
    )

    target_set = set(
        target_players
    )

    coverage = (
        consensus.groupby(
            [
                "CAREER_PHASE",
                "SIDE",
                "DIMENSION",
            ],
            dropna=False,
        )[
            "PLAYER_NAME"
        ]
        .apply(
            lambda values: len(
                set(values)
                & target_set
            )
        )
        .rename(
            "PLAYER_COVERAGE_COUNT"
        )
        .reset_index()
    )

    consensus = consensus.merge(
        coverage,
        on=[
            "CAREER_PHASE",
            "SIDE",
            "DIMENSION",
        ],
        how="left",
        validate="many_to_one",
    )

    def evidence_status(
        row: pd.Series,
    ) -> str:
        if not bool(
            row[
                "DIMENSION_PRIMARY_ELIGIBLE"
            ]
        ):
            return "supplementary_dimension"

        if (
            str(
                row["CAREER_PHASE"]
            )
            != "career"
        ):
            return "phase_detail_only"

        if int(
            row["SOURCE_FAMILIES"]
        ) < int(
            row[
                "MIN_SOURCE_FAMILIES"
            ]
        ):
            return (
                "insufficient_source_families"
            )

        if int(
            row["TIER_A_FAMILIES"]
        ) < int(
            row[
                "MIN_TIER_A_FAMILIES"
            ]
        ):
            return "missing_tier_a_source"

        if int(
            row[
                "PLAYER_COVERAGE_COUNT"
            ]
        ) < len(
            target_players
        ):
            return (
                "missing_comparison_player"
            )

        return "eligible"

    consensus[
        "EVIDENCE_STATUS"
    ] = consensus.apply(
        evidence_status,
        axis=1,
    )

    consensus[
        "PRIMARY_MODEL_ELIGIBLE"
    ] = consensus[
        "EVIDENCE_STATUS"
    ].eq(
        "eligible"
    )

    return (
        consensus[
            CONSENSUS_COLUMNS
        ]
        .sort_values(
            [
                "SIDE",
                "DIMENSION",
                "CAREER_PHASE",
                "PLAYER_NAME",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_expert_release_blockers(
    sources: pd.DataFrame,
    claims: pd.DataFrame,
    dimensions: pd.DataFrame,
    consensus: pd.DataFrame,
    target_players: tuple[str, ...] = (
        TARGET_PLAYERS
    ),
) -> pd.DataFrame:
    rows: list[
        dict[str, str]
    ] = []

    if sources.empty:
        rows.append(
            {
                "BLOCKER_TYPE": (
                    "missing_sources"
                ),
                "SIDE": "",
                "DIMENSION": "",
                "PLAYER_NAME": "",
                "DETAIL": (
                    "No expert sources have "
                    "been registered."
                ),
            }
        )

    if claims.empty:
        rows.append(
            {
                "BLOCKER_TYPE": (
                    "missing_claims"
                ),
                "SIDE": "",
                "DIMENSION": "",
                "PLAYER_NAME": "",
                "DETAIL": (
                    "No expert claims have "
                    "been verified."
                ),
            }
        )

    primary = dimensions.copy()

    primary[
        "PRIMARY_ELIGIBLE"
    ] = _coerce_bool(
        primary[
            "PRIMARY_ELIGIBLE"
        ],
        "PRIMARY_ELIGIBLE",
    )

    primary = primary[
        primary[
            "PRIMARY_ELIGIBLE"
        ]
    ]

    for dimension_row in primary.itertuples(
        index=False
    ):
        side = str(
            dimension_row.SIDE
        ).strip().casefold()

        dimension = str(
            dimension_row.DIMENSION
        ).strip()

        for player_name in target_players:
            matching = consensus[
                consensus[
                    "PLAYER_NAME"
                ].eq(
                    player_name
                )
                & consensus[
                    "CAREER_PHASE"
                ].eq(
                    "career"
                )
                & consensus[
                    "SIDE"
                ].eq(
                    side
                )
                & consensus[
                    "DIMENSION"
                ].eq(
                    dimension
                )
            ]

            if matching.empty:
                rows.append(
                    {
                        "BLOCKER_TYPE": (
                            "missing_dimension_evidence"
                        ),
                        "SIDE": side,
                        "DIMENSION": dimension,
                        "PLAYER_NAME": player_name,
                        "DETAIL": (
                            "No career-level "
                            "consensus row exists."
                        ),
                    }
                )

                continue

            row = matching.iloc[0]

            if not bool(
                row[
                    "PRIMARY_MODEL_ELIGIBLE"
                ]
            ):
                rows.append(
                    {
                        "BLOCKER_TYPE": (
                            "ineligible_dimension_evidence"
                        ),
                        "SIDE": side,
                        "DIMENSION": dimension,
                        "PLAYER_NAME": player_name,
                        "DETAIL": str(
                            row[
                                "EVIDENCE_STATUS"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "BLOCKER_TYPE",
            "SIDE",
            "DIMENSION",
            "PLAYER_NAME",
            "DETAIL",
        ],
    )
