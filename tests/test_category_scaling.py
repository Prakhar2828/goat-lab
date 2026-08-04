from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goatlab.models.category_scaling import (
    SCALING_SCENARIOS,
    build_saturation_audit,
    build_scaling_comparison,
    percentile_to_bounded_logit_score,
    percentile_to_normal_score,
    reference_percentile,
    robust_mad_reference_score,
    transform_percentile_scores,
)


def test_percentile_transforms_are_monotonic_bounded_and_finite() -> None:
    percentiles = pd.Series(
        [
            0.0,
            1.0,
            25.0,
            50.0,
            75.0,
            99.0,
            100.0,
        ]
    )

    for transform in (
        percentile_to_normal_score,
        percentile_to_bounded_logit_score,
    ):
        result = transform(
            percentiles
        )

        assert np.isfinite(
            result
        ).all()
        assert result.between(
            0.0,
            100.0,
            inclusive="both",
        ).all()
        assert result.is_monotonic_increasing


def test_tail_transforms_expand_elite_percentile_gap() -> None:
    values = pd.Series(
        [
            98.8,
            99.7,
        ]
    )
    original_gap = float(
        values.iloc[1]
        - values.iloc[0]
    )

    for transform in (
        percentile_to_normal_score,
        percentile_to_bounded_logit_score,
    ):
        transformed = transform(
            values
        )
        transformed_gap = float(
            transformed.iloc[1]
            - transformed.iloc[0]
        )

        assert transformed_gap > (
            original_gap
        )


def test_robust_reference_score_is_monotonic_and_has_fallback() -> None:
    reference = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            100.0,
        ]
    )

    low, _, _, low_method = (
        robust_mad_reference_score(
            2.0,
            reference,
        )
    )
    high, _, _, high_method = (
        robust_mad_reference_score(
            4.0,
            reference,
        )
    )

    assert high > low
    assert low_method == "median_mad"
    assert high_method == "median_mad"

    constant_score, _, scale, method = (
        robust_mad_reference_score(
            5.0,
            pd.Series(
                [
                    5.0,
                    5.0,
                    5.0,
                ]
            ),
        )
    )

    assert np.isclose(
        constant_score,
        50.0,
    )
    assert np.isclose(
        scale,
        1.0,
    )
    assert (
        method
        == "constant_reference_fallback"
    )


def test_reference_percentile_handles_missing_values() -> None:
    reference = pd.Series(
        [
            1.0,
            2.0,
            np.nan,
            3.0,
        ]
    )

    assert np.isclose(
        reference_percentile(
            2.0,
            reference,
        ),
        66.66666666666667,
    )
    assert np.isnan(
        reference_percentile(
            np.nan,
            reference,
        )
    )


def test_transform_rejects_raw_reference_scenario_without_reference() -> None:
    with pytest.raises(
        ValueError,
        match="requires raw reference",
    ):
        transform_percentile_scores(
            pd.Series(
                [
                    99.0,
                ]
            ),
            "robust_mad_reference",
        )


def test_scaling_comparison_is_complete_and_preserves_native_scales() -> None:
    category_scores = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player A",
                "peak": 99.0,
                "prime": 98.0,
                "longevity": 97.0,
                "regular_season": 96.0,
                "playoffs": 95.0,
                "offense": 94.0,
                "defense": 93.0,
                "winning_context": 60.0,
                "cultural_impact": 80.0,
            },
            {
                "PLAYER_NAME": "Player B",
                "peak": 98.0,
                "prime": 97.0,
                "longevity": 96.0,
                "regular_season": 95.0,
                "playoffs": 94.0,
                "offense": 93.0,
                "defense": 92.0,
                "winning_context": 55.0,
                "cultural_impact": 85.0,
            },
        ]
    )

    reference_rows = []
    for player_index in range(
        1,
        11,
    ):
        row = {
            "PLAYER_NAME": (
                "Player A"
                if player_index == 9
                else (
                    "Player B"
                    if player_index == 8
                    else f"Reference {player_index}"
                )
            )
        }
        for raw_column in (
            "peak_raw",
            "prime_raw",
            "longevity_raw",
            "regular_season_raw",
            "playoffs_raw",
            "offense_raw",
            "defense_raw",
        ):
            row[raw_column] = float(
                player_index
            )
        reference_rows.append(
            row
        )

    comparison = (
        build_scaling_comparison(
            category_scores,
            pd.DataFrame(
                reference_rows
            ),
        )
    )

    assert len(comparison) == (
        len(SCALING_SCENARIOS)
        * 2
        * 9
    )
    assert set(
        comparison[
            "SCENARIO"
        ]
    ) == set(
        SCALING_SCENARIOS
    )
    assert comparison[
        "SCORE"
    ].between(
        0.0,
        100.0,
        inclusive="both",
    ).all()

    native = comparison[
        comparison[
            "CATEGORY"
        ].eq(
            "winning_context"
        )
        & comparison[
            "PLAYER_NAME"
        ].eq(
            "Player A"
        )
    ]

    assert set(
        native[
            "SCORE"
        ]
    ) == {
        60.0,
    }


def test_saturation_audit_flags_compressed_historical_tail() -> None:
    comparison = pd.DataFrame(
        [
            {
                "SCENARIO": "historical_percentile",
                "PLAYER_NAME": "A",
                "CATEGORY": "peak",
                "SCORE": 99.7,
            },
            {
                "SCENARIO": "historical_percentile",
                "PLAYER_NAME": "B",
                "CATEGORY": "peak",
                "SCORE": 98.9,
            },
            {
                "SCENARIO": "expanded",
                "PLAYER_NAME": "A",
                "CATEGORY": "peak",
                "SCORE": 96.0,
            },
            {
                "SCENARIO": "expanded",
                "PLAYER_NAME": "B",
                "CATEGORY": "peak",
                "SCORE": 90.0,
            },
        ]
    )

    audit = build_saturation_audit(
        comparison
    ).set_index(
        [
            "SCENARIO",
            "CATEGORY",
        ]
    )

    assert bool(
        audit.loc[
            (
                "historical_percentile",
                "peak",
            ),
            "SATURATION_FLAG",
        ]
    )
    assert not bool(
        audit.loc[
            (
                "expanded",
                "peak",
            ),
            "SATURATION_FLAG",
        ]
    )
