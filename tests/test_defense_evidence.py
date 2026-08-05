from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goatlab.models.defense_evidence import (
    AWARD_SCORE_BENCHMARK_POINTS,
    build_awards_scores,
    build_defense_evidence_scores,
    build_film_diagnostics,
    classify_defensive_award,
    normalize_defensive_awards,
)


def _awards_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PERSON_ID": 893,
                "FIRST_NAME": "Michael",
                "LAST_NAME": "Jordan",
                "DESCRIPTION": (
                    "NBA Defensive Player of the Year"
                ),
                "SEASON": "1987-88",
                "ALL_NBA_TEAM_NUMBER": np.nan,
            },
            {
                "PERSON_ID": 893,
                "FIRST_NAME": "Michael",
                "LAST_NAME": "Jordan",
                "DESCRIPTION": (
                    "NBA All-Defensive First Team"
                ),
                "SEASON": "1987-88",
                "ALL_NBA_TEAM_NUMBER": 1,
            },
            {
                "PERSON_ID": 893,
                "FIRST_NAME": "Michael",
                "LAST_NAME": "Jordan",
                "DESCRIPTION": (
                    "NBA All-Defensive First Team"
                ),
                "SEASON": "1987-88",
                "ALL_NBA_TEAM_NUMBER": 1,
            },
            {
                "PERSON_ID": 2544,
                "FIRST_NAME": "LeBron",
                "LAST_NAME": "James",
                "DESCRIPTION": (
                    "NBA All-Defensive First Team"
                ),
                "SEASON": "2011-12",
                "ALL_NBA_TEAM_NUMBER": 1,
            },
            {
                "PERSON_ID": 2544,
                "FIRST_NAME": "LeBron",
                "LAST_NAME": "James",
                "DESCRIPTION": (
                    "NBA All-Defensive Second Team"
                ),
                "SEASON": "2013-14",
                "ALL_NBA_TEAM_NUMBER": 2,
            },
            {
                "PERSON_ID": 2544,
                "FIRST_NAME": "LeBron",
                "LAST_NAME": "James",
                "DESCRIPTION": "Most Valuable Player",
                "SEASON": "2011-12",
                "ALL_NBA_TEAM_NUMBER": np.nan,
            },
        ]
    )


def _film_fixture(
    primary: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Michael Jordan",
                "SIDE": "defense",
                "DIMENSION": "ball_denial",
                "CONSENSUS_SCORE": 88.0,
                "CONSENSUS_LOW": 80.0,
                "CONSENSUS_HIGH": 92.0,
                "SOURCE_FAMILIES": 1,
                "DEFAULT_WEIGHT": 1.0,
                "PRIMARY_MODEL_ELIGIBLE": primary,
            },
            {
                "PLAYER_NAME": "LeBron James",
                "SIDE": "defense",
                "DIMENSION": "help_positioning",
                "CONSENSUS_SCORE": 86.0,
                "CONSENSUS_LOW": 78.0,
                "CONSENSUS_HIGH": 90.0,
                "SOURCE_FAMILIES": 1,
                "DEFAULT_WEIGHT": 1.0,
                "PRIMARY_MODEL_ELIGIBLE": primary,
            },
        ]
    )


@pytest.mark.parametrize(
    ("description", "team_number", "expected"),
    [
        (
            "NBA Defensive Player of the Year",
            np.nan,
            "defensive_player_of_year",
        ),
        (
            "NBA All-Defensive First Team",
            1,
            "all_defensive_first",
        ),
        (
            "NBA All Defensive Second Team",
            2,
            "all_defensive_second",
        ),
        (
            "NBA All-Defensive Team",
            np.nan,
            "all_defensive_generic",
        ),
        (
            "Most Valuable Player",
            np.nan,
            None,
        ),
    ],
)
def test_classify_defensive_award(
    description: str,
    team_number: float,
    expected: str | None,
) -> None:
    assert (
        classify_defensive_award(
            description,
            team_number,
        )
        == expected
    )


def test_normalize_defensive_awards_filters_and_deduplicates() -> None:
    normalized = normalize_defensive_awards(
        _awards_fixture()
    )

    assert len(normalized) == 4
    assert normalized[
        [
            "PLAYER_NAME",
            "SEASON",
            "AWARD_KEY",
        ]
    ].duplicated().sum() == 0

    assert set(
        normalized["AWARD_KEY"]
    ) == {
        "defensive_player_of_year",
        "all_defensive_first",
        "all_defensive_second",
    }


def test_awards_scores_use_fixed_benchmark() -> None:
    normalized = normalize_defensive_awards(
        _awards_fixture()
    )
    scores = build_awards_scores(
        normalized
    ).set_index(
        "PLAYER_NAME"
    )

    jordan_points = 7.0
    lebron_points = 3.0

    assert np.isclose(
        scores.loc[
            "Michael Jordan",
            "defense_awards_score",
        ],
        100
        * jordan_points
        / AWARD_SCORE_BENCHMARK_POINTS,
    )

    assert np.isclose(
        scores.loc[
            "LeBron James",
            "defense_awards_score",
        ],
        100
        * lebron_points
        / AWARD_SCORE_BENCHMARK_POINTS,
    )


def test_partial_phase_film_remains_diagnostic() -> None:
    diagnostics = build_film_diagnostics(
        _film_fixture(
            primary=False
        )
    )

    assert not diagnostics[
        "FILM_USED_IN_MODEL"
    ].any()

    assert diagnostics[
        "defense_film_score"
    ].isna().all()

    assert diagnostics[
        "DEFENSE_FILM_PRIMARY_ROWS"
    ].eq(0).all()


def test_primary_eligible_film_can_produce_score() -> None:
    diagnostics = (
        build_film_diagnostics(
            _film_fixture(
                primary=True
            )
        )
        .set_index(
            "PLAYER_NAME"
        )
    )

    assert diagnostics[
        "FILM_USED_IN_MODEL"
    ].all()

    assert np.isclose(
        diagnostics.loc[
            "Michael Jordan",
            "defense_film_score",
        ],
        88.0,
    )

    assert np.isclose(
        diagnostics.loc[
            "LeBron James",
            "defense_film_score",
        ],
        86.0,
    )


def test_combined_scores_do_not_turn_missing_film_into_zero() -> None:
    scores, normalized = (
        build_defense_evidence_scores(
            _awards_fixture(),
            _film_fixture(
                primary=False
            ),
        )
    )

    assert len(normalized) == 4
    assert len(scores) == 2

    assert scores[
        "defense_film_score"
    ].isna().all()

    assert scores[
        "AWARDS_USED_IN_MODEL"
    ].all()

    assert not scores[
        "FILM_USED_IN_MODEL"
    ].any()

    assert scores[
        "DEFENSE_EVIDENCE_STATUS"
    ].eq(
        "awards_with_film_diagnostic_only"
    ).all()

    assert not scores[
        "DEFENSE_RELEASE_BLOCKER"
    ].any()
