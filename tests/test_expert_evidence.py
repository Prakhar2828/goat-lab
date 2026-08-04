from __future__ import annotations

import pandas as pd
import pytest

from goatlab.models.expert_evidence import (
    build_expert_consensus,
    score_expert_sources,
    validate_expert_evidence,
)


def _dimensions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SIDE": "offense",
                "DIMENSION": "passing_vision",
                "DESCRIPTION": "Passing vision",
                "MIN_SOURCE_FAMILIES": 3,
                "MIN_TIER_A_FAMILIES": 1,
                "PRIMARY_ELIGIBLE": True,
                "DEFAULT_WEIGHT": 1.0,
            }
        ]
    )


def _source(
    source_id: str,
    family: str,
    tier: str,
) -> dict[str, object]:
    scores = {
        "A": [
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
        ],
        "B": [
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
        ],
        "C": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
        ],
    }[tier]

    return {
        "SOURCE_ID": source_id,
        "SOURCE_FAMILY": family,
        "ANALYST": family,
        "PUBLICATION": "Publication",
        "TITLE": source_id,
        "PUBLICATION_DATE": "2020-01-01",
        "URL": f"https://example.com/{source_id}",
        "SOURCE_TYPE": "film_analysis",
        "EXPERTISE_SCORE": scores[0],
        "FILM_SPECIFICITY_SCORE": scores[1],
        "METHODOLOGY_SCORE": scores[2],
        "SAMPLE_DISCLOSURE_SCORE": scores[3],
        "STATISTICAL_SUPPORT_SCORE": scores[4],
        "CAREER_COVERAGE_SCORE": scores[5],
        "BALANCED_EVIDENCE_SCORE": scores[6],
        "INDEPENDENCE_SCORE": scores[7],
        "NOTES": "",
    }


def _claim(
    claim_id: str,
    source_id: str,
    player_name: str,
    direction: str = "strength",
    status: str = "verified",
) -> dict[str, object]:
    return {
        "CLAIM_ID": claim_id,
        "SOURCE_ID": source_id,
        "PLAYER_NAME": player_name,
        "CAREER_PHASE": "career",
        "SEASON_START": "",
        "SEASON_END": "",
        "SEASON_TYPE": "all",
        "SIDE": "offense",
        "DIMENSION": "passing_vision",
        "CLAIM_DIRECTION": direction,
        "CLAIM_STRENGTH": 3,
        "EVIDENCE_TYPE": "film_analysis",
        "FILM_EXAMPLES_PRESENT": True,
        "SAMPLE_SIZE_DISCLOSED": True,
        "CONFIDENCE": 1.0,
        "SUPPORTING_LOCATION": "Section 1",
        "SUMMARY": "Supported analytical claim.",
        "LIMITATIONS": "",
        "REVIEW_STATUS": status,
    }


def test_source_quality_tiers() -> None:
    sources = pd.DataFrame(
        [
            _source(
                "A1",
                "Family A",
                "A",
            ),
            _source(
                "B1",
                "Family B",
                "B",
            ),
            _source(
                "C1",
                "Family C",
                "C",
            ),
        ]
    )

    scored = (
        score_expert_sources(
            sources
        )
        .set_index(
            "SOURCE_ID"
        )
    )

    assert (
        scored.loc[
            "A1",
            "SOURCE_TIER",
        ]
        == "A"
    )

    assert (
        scored.loc[
            "B1",
            "SOURCE_TIER",
        ]
        == "B"
    )

    assert (
        scored.loc[
            "C1",
            "SOURCE_TIER",
        ]
        == "C"
    )


def test_validation_rejects_out_of_range_source_score() -> None:
    sources = pd.DataFrame(
        [
            _source(
                "A1",
                "Family A",
                "A",
            )
        ]
    )

    sources.loc[
        0,
        "EXPERTISE_SCORE",
    ] = 4

    claims = pd.DataFrame(
        [
            _claim(
                "C1",
                "A1",
                "Michael Jordan",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="EXPERTISE_SCORE",
    ):
        validate_expert_evidence(
            sources,
            claims,
            _dimensions(),
        )


def test_validation_rejects_unknown_source() -> None:
    sources = pd.DataFrame(
        [
            _source(
                "A1",
                "Family A",
                "A",
            )
        ]
    )

    claims = pd.DataFrame(
        [
            _claim(
                "C1",
                "UNKNOWN",
                "Michael Jordan",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="unknown sources",
    ):
        validate_expert_evidence(
            sources,
            claims,
            _dimensions(),
        )


def test_source_family_count_is_not_article_count() -> None:
    sources = pd.DataFrame(
        [
            _source(
                "A1",
                "Family A",
                "A",
            ),
            _source(
                "A2",
                "Family A",
                "A",
            ),
            _source(
                "B1",
                "Family B",
                "B",
            ),
            _source(
                "C1",
                "Family C",
                "B",
            ),
        ]
    )

    claims: list[
        dict[str, object]
    ] = []

    for player_name in [
        "Michael Jordan",
        "LeBron James",
    ]:
        for source_id in [
            "A1",
            "A2",
            "B1",
            "C1",
        ]:
            claims.append(
                _claim(
                    (
                        f"{player_name}-"
                        f"{source_id}"
                    ),
                    source_id,
                    player_name,
                )
            )

    consensus = (
        build_expert_consensus(
            sources,
            pd.DataFrame(claims),
            _dimensions(),
        )
    )

    assert set(
        consensus[
            "SOURCE_FAMILIES"
        ]
    ) == {
        3,
    }


def test_primary_eligibility_requires_both_players() -> None:
    sources = pd.DataFrame(
        [
            _source(
                "A1",
                "Family A",
                "A",
            ),
            _source(
                "B1",
                "Family B",
                "B",
            ),
            _source(
                "C1",
                "Family C",
                "B",
            ),
        ]
    )

    claims: list[
        dict[str, object]
    ] = []

    for player_name in [
        "Michael Jordan",
        "LeBron James",
    ]:
        for source_id in [
            "A1",
            "B1",
            "C1",
        ]:
            claims.append(
                _claim(
                    (
                        f"{player_name}-"
                        f"{source_id}"
                    ),
                    source_id,
                    player_name,
                )
            )

    consensus = (
        build_expert_consensus(
            sources,
            pd.DataFrame(claims),
            _dimensions(),
        )
    )

    assert consensus[
        "PRIMARY_MODEL_ELIGIBLE"
    ].all()

    lebron_missing = pd.DataFrame(
        [
            claim
            for claim in claims
            if claim[
                "PLAYER_NAME"
            ]
            != "LeBron James"
        ]
    )

    incomplete = (
        build_expert_consensus(
            sources,
            lebron_missing,
            _dimensions(),
        )
    )

    assert not incomplete[
        "PRIMARY_MODEL_ELIGIBLE"
    ].any()

    assert set(
        incomplete[
            "EVIDENCE_STATUS"
        ]
    ) == {
        "missing_comparison_player",
    }


def test_rejected_claims_do_not_enter_consensus() -> None:
    sources = pd.DataFrame(
        [
            _source(
                "A1",
                "Family A",
                "A",
            )
        ]
    )

    claims = pd.DataFrame(
        [
            _claim(
                "C1",
                "A1",
                "Michael Jordan",
                status="rejected",
            )
        ]
    )

    consensus = (
        build_expert_consensus(
            sources,
            claims,
            _dimensions(),
        )
    )

    assert consensus.empty
