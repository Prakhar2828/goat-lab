from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path("scripts/import_lowe_claims.py")

EXPECTED_COUNTS = {
    ("LOWE_GOAT_DEBATE_2020", "LeBron James", "defense"): 2,
    ("LOWE_GOAT_DEBATE_2020", "LeBron James", "offense"): 5,
    ("LOWE_GOAT_DEBATE_2020", "Michael Jordan", "offense"): 4,
    ("LOWE_LAKERS_DEFENSE_2020", "LeBron James", "defense"): 6,
}


def _load_claims() -> pd.DataFrame:
    namespace = runpy.run_path(
        str(SCRIPT_PATH),
        run_name="lowe_claims_test",
    )

    build_claims = namespace.get(
        "build_claims"
    )

    assert callable(build_claims)

    claims = build_claims()

    assert isinstance(claims, list)

    return pd.DataFrame(claims)


def test_lowe_claim_inventory() -> None:
    claims = _load_claims()

    assert len(claims) == 17
    assert claims["CLAIM_ID"].is_unique

    counts = (
        claims.groupby(
            [
                "SOURCE_ID",
                "PLAYER_NAME",
                "SIDE",
            ]
        )
        .size()
        .to_dict()
    )

    assert counts == EXPECTED_COUNTS


def test_lowe_claim_safeguards() -> None:
    claims = _load_claims()

    assert set(claims["SOURCE_ID"]) == {
        "LOWE_GOAT_DEBATE_2020",
        "LOWE_LAKERS_DEFENSE_2020",
    }

    assert set(claims["PLAYER_NAME"]) <= {
        "Michael Jordan",
        "LeBron James",
    }

    assert set(claims["SIDE"]) <= {
        "offense",
        "defense",
    }

    assert not claims["CAREER_PHASE"].eq(
        "career"
    ).any()

    assert claims["REVIEW_STATUS"].eq(
        "verified_with_qualification"
    ).all()

    for column, minimum_length in {
        "SUPPORTING_LOCATION": 12,
        "SUMMARY": 20,
        "LIMITATIONS": 20,
    }.items():
        lengths = (
            claims[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.len()
        )

        assert lengths.ge(
            minimum_length
        ).all()
