from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from goatlab.models.category_hierarchy import (
    CATEGORIES,
    build_overlap_audit,
    build_weight_table,
    load_hierarchy_config,
    score_category_hierarchy,
)


def _config() -> dict:
    return load_hierarchy_config(
        "configs/category_hierarchy.json"
    )


def test_hierarchy_covers_each_category_once() -> None:
    config = _config()
    categories = [
        item["name"]
        for group in config["groups"]
        for item in group["categories"]
    ]

    assert len(categories) == len(set(categories))
    assert set(categories) == set(CATEGORIES)


def test_group_caps_sum_to_one() -> None:
    config = _config()

    assert np.isclose(
        sum(float(group["cap"]) for group in config["groups"]),
        1.0,
    )


def test_within_group_weights_sum_to_one() -> None:
    config = _config()

    for group in config["groups"]:
        assert np.isclose(
            sum(
                float(item["within_group_weight"])
                for item in group["categories"]
            ),
            1.0,
        )


def test_provisional_total_weights_sum_to_one() -> None:
    weights = build_weight_table(_config())

    assert np.isclose(
        weights["PROVISIONAL_TOTAL_WEIGHT"].sum(),
        1.0,
    )


def test_complete_hierarchy_score_matches_weights() -> None:
    config = _config()
    frame = pd.DataFrame(
        [
            {
                "PLAYER_NAME": "Player A",
                **{
                    category: float(index * 10)
                    for index, category in enumerate(
                        CATEGORIES,
                        start=1,
                    )
                },
            }
        ]
    )

    result = score_category_hierarchy(frame, config)
    weights = (
        build_weight_table(config)
        .set_index("CATEGORY")[
            "PROVISIONAL_TOTAL_WEIGHT"
        ]
    )

    expected = sum(
        float(frame.iloc[0][category])
        * float(weights[category])
        for category in CATEGORIES
    )

    assert np.isclose(
        result.loc[0, "HIERARCHICAL_SCORE"],
        expected,
    )
    assert np.isclose(
        result.loc[0, "HIERARCHICAL_COVERAGE"],
        1.0,
    )


def test_missing_evidence_is_reweighted_not_zeroed() -> None:
    config = _config()
    row = {
        "PLAYER_NAME": "Player A",
        **{
            category: 80.0
            for category in CATEGORIES
        },
    }
    row["cultural_impact"] = np.nan

    result = score_category_hierarchy(
        pd.DataFrame([row]),
        config,
    )

    assert np.isclose(
        result.loc[0, "HIERARCHICAL_SCORE"],
        80.0,
    )
    assert np.isclose(
        result.loc[0, "HIERARCHICAL_COVERAGE"],
        0.90,
    )


def _reference_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = np.linspace(0, 10, rows)

    return pd.DataFrame(
        {
            "peak_raw": base,
            "prime_raw": base * 1.2,
            "longevity_raw": rng.normal(size=rows),
            "regular_season_raw": (
                base
                + rng.normal(scale=0.1, size=rows)
            ),
            "playoffs_raw": rng.normal(size=rows),
            "offense_raw": (
                base * 0.8
                + rng.normal(scale=0.1, size=rows)
            ),
            "defense_raw": rng.normal(size=rows),
        }
    )


def test_overlap_audit_has_all_reference_pairs() -> None:
    config = _config()
    audit = build_overlap_audit(
        _reference_frame(),
        config,
    )

    assert len(audit) == len(
        list(
            combinations(
                config["reference_columns"],
                2,
            )
        )
    )
    assert len(audit) == 21
    assert not audit["RELEASE_BLOCKER"].any()


def test_overlap_audit_distinguishes_declared_dependency() -> None:
    config = _config()
    audit = build_overlap_audit(
        _reference_frame(),
        config,
    )

    peak_prime = audit[
        (
            audit["CATEGORY_A"].eq("peak")
            & audit["CATEGORY_B"].eq("prime")
        )
        |
        (
            audit["CATEGORY_A"].eq("prime")
            & audit["CATEGORY_B"].eq("peak")
        )
    ].iloc[0]

    assert peak_prime["STATUS"] == "declared_dependency"
    assert bool(peak_prime["DECLARED_DEPENDENCY"])
