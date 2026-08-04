from __future__ import annotations

import pandas as pd

from goatlab.models.expert_evidence import (
    validate_expert_evidence,
)
from goatlab.settings import settings
from scripts.import_goldsberry_claims import (
    SOURCE_IDS,
    build_claims,
)


def test_goldsberry_claim_inventory() -> None:
    claims = build_claims()

    assert len(claims) == 15
    assert set(
        claims[
            "SOURCE_ID"
        ]
    ) == SOURCE_IDS
    assert set(
        claims[
            "SIDE"
        ]
    ) == {
        "offense",
    }
    assert not claims[
        "CAREER_PHASE"
    ].eq(
        "career"
    ).any()
    assert not claims[
        "CLAIM_ID"
    ].duplicated().any()

    counts = (
        claims.groupby(
            "SOURCE_ID"
        )
        .size()
        .to_dict()
    )

    assert counts == {
        "GOLD_LBJ_ATLAS_2018": 6,
        "GOLD_LBJ_SCORING_RECORD_2023": 3,
        "GOLD_MJ_SCORING_2020": 6,
    }


def test_goldsberry_claims_validate_against_registry() -> None:
    sources = pd.read_csv(
        settings.manual_dir
        / "expert_sources.csv"
    )
    dimensions = pd.read_csv(
        settings.manual_dir
        / "expert_analysis_dimensions.csv"
    )
    claims = build_claims()

    validate_expert_evidence(
        sources,
        claims,
        dimensions,
    )

    family_count = (
        sources.loc[
            sources[
                "SOURCE_ID"
            ].isin(
                SOURCE_IDS
            ),
            "SOURCE_FAMILY",
        ]
        .nunique()
    )

    assert family_count == 1
