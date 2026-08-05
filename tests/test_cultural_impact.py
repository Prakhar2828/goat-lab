from __future__ import annotations

import numpy as np
import pandas as pd

from goatlab.models.cultural_impact import (
    calculate_attention_scores,
    calculate_rubric_scores,
    combine_cultural_scores,
)

DIMENSIONS = [
    "commercial_global_reach",
    "basketball_culture_influence",
    "media_entertainment_reach",
    "philanthropy_social_institutions",
]


def test_attention_scores_sum_to_100() -> None:
    rows = []

    for player, base in [
        ("Player A", 100),
        ("Player B", 80),
    ]:
        for date in pd.date_range(
            "2020-01-01",
            periods=730,
            freq="D",
        ):
            rows.append(
                {
                    "PLAYER_NAME": player,
                    "date": date,
                    "views": base,
                }
            )

    scores = calculate_attention_scores(
        pd.DataFrame(rows)
    )

    assert np.isclose(
        scores["ATTENTION_SCORE"].sum(),
        100.0,
    )

    assert (
        scores["ATTENTION_SCORE"]
        .between(0, 100)
        .all()
    )


def test_complete_rubric_builds_score() -> None:
    rows = []

    for player, score in [
        ("Player A", 80),
        ("Player B", 70),
    ]:
        for dimension in DIMENSIONS:
            rows.append(
                {
                    "PLAYER_NAME": player,
                    "DIMENSION": dimension,
                    "SCORE_0_100": score,
                    "CONFIDENCE": "High",
                    "SOURCE_IDS": "SRC001",
                    "RATIONALE": "Test evidence",
                }
            )

    result = calculate_rubric_scores(
        pd.DataFrame(rows)
    )

    assert result[
        "RUBRIC_COMPLETE"
    ].all()

    assert result[
        "RUBRIC_SCORE"
    ].notna().all()

    assert np.isclose(
        result.loc[
            result["PLAYER_NAME"]
            == "Player A",
            "RUBRIC_SCORE",
        ].iloc[0],
        80.0,
    )


def test_incomplete_rubric_blocks_final_score() -> None:
    attention = pd.DataFrame(
        {
            "PLAYER_NAME": [
                "Player A",
            ],
            "ATTENTION_SCORE": [
                55.0,
            ],
        }
    )

    rubric = pd.DataFrame(
        {
            "PLAYER_NAME": [
                "Player A",
            ],
            "RUBRIC_SCORE": [
                np.nan,
            ],
            "RUBRIC_COVERAGE": [
                0.5,
            ],
            "RUBRIC_CONFIDENCE": [
                1.0,
            ],
            "RUBRIC_COMPLETE": [
                False,
            ],
            "MISSING_DIMENSIONS": [
                "missing",
            ],
        }
    )

    result = combine_cultural_scores(
        attention,
        rubric,
    )

    assert pd.isna(
        result.loc[
            0,
            "cultural_impact_raw",
        ]
    )

    assert not bool(
        result.loc[
            0,
            "CULTURAL_SCORE_COMPLETE",
        ]
    )
