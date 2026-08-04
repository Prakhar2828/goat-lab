from __future__ import annotations

import pandas as pd
import pytest

from goatlab.data.playoff_rounds import (
    add_canonical_playoff_rounds,
    normalize_round_number,
)


def _append_series(
    rows: list[dict[str, object]],
    *,
    series_id: str,
    winner: int,
    loser: int,
    start: str,
    end: str,
    label: str = "",
) -> None:
    rows.extend(
        [
            {
                "SERIES_ID": series_id,
                "SEASON": "2020-21",
                "TEAM_ID": winner,
                "OPP_TEAM_ID": loser,
                "TEAM_WON_SERIES": 1,
                "SERIES_START_DATE": start,
                "SERIES_END_DATE": end,
                "ROUND": label,
            },
            {
                "SERIES_ID": series_id,
                "SEASON": "2020-21",
                "TEAM_ID": loser,
                "OPP_TEAM_ID": winner,
                "TEAM_WON_SERIES": 0,
                "SERIES_START_DATE": start,
                "SERIES_END_DATE": end,
                "ROUND": label,
            },
        ]
    )


def _full_bracket() -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for index in range(8):
        _append_series(
            rows,
            series_id=f"R1-{index}",
            winner=index + 1,
            loser=index + 9,
            start="2021-04-01",
            end="2021-04-05",
            label=(
                "East - First Round"
                if index == 0
                else ""
            ),
        )

    for index in range(4):
        _append_series(
            rows,
            series_id=f"R2-{index}",
            winner=index + 1,
            loser=index + 5,
            start="2021-05-01",
            end="2021-05-05",
            label=(
                "West Conf. Semifinals"
                if index == 0
                else ""
            ),
        )

    _append_series(
        rows,
        series_id="R3-0",
        winner=1,
        loser=3,
        start="2021-06-01",
        end="2021-06-05",
    )

    _append_series(
        rows,
        series_id="R3-1",
        winner=2,
        loser=4,
        start="2021-06-01",
        end="2021-06-05",
    )

    _append_series(
        rows,
        series_id="R4-0",
        winner=1,
        loser=2,
        start="2021-07-01",
        end="2021-07-05",
        label="NBA Finals",
    )

    return pd.DataFrame(rows)


def test_round_label_normalization() -> None:
    assert normalize_round_number(
        "East - First Round"
    ) == 1

    assert normalize_round_number(
        "West Conf. Semifinals"
    ) == 2

    assert normalize_round_number(
        "East - Conf. Finals"
    ) == 3

    assert normalize_round_number(
        "NBA Finals"
    ) == 4


def test_missing_rounds_are_inferred() -> None:
    result = add_canonical_playoff_rounds(
        _full_bracket()
    )

    series = result.drop_duplicates(
        "SERIES_ID"
    )

    counts = (
        series[
            "ROUND_NUMBER"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    assert counts == {
        1: 8,
        2: 4,
        3: 2,
        4: 1,
    }

    assert not result[
        "ROUND_NUMBER"
    ].isna().any()

    assert set(
        result["ROUND"]
    ) == {
        "First Round",
        "Conference Semifinals",
        "Conference Finals",
        "NBA Finals",
    }


def test_conflicting_source_label_fails() -> None:
    frame = _full_bracket()

    frame.loc[
        frame["SERIES_ID"].eq(
            "R4-0"
        ),
        "ROUND",
    ] = "First Round"

    with pytest.raises(
        ValueError,
        match="conflict",
    ):
        add_canonical_playoff_rounds(
            frame
        )


def test_round_inference_survives_shuffled_rows() -> None:
    frame = (
        _full_bracket()
        .sample(
            frac=1,
            random_state=23,
        )
        .reset_index(
            drop=True
        )
    )

    result = add_canonical_playoff_rounds(
        frame
    )

    series = (
        result.drop_duplicates(
            "SERIES_ID"
        )
        .set_index(
            "SERIES_ID"
        )
    )

    assert (
        series.loc[
            "R1-0",
            "ROUND_NUMBER",
        ]
        == 1
    )

    assert (
        series.loc[
            "R2-0",
            "ROUND_NUMBER",
        ]
        == 2
    )

    assert (
        series.loc[
            "R3-0",
            "ROUND_NUMBER",
        ]
        == 3
    )

    assert (
        series.loc[
            "R4-0",
            "ROUND_NUMBER",
        ]
        == 4
    )

    counts = (
        series[
            "ROUND_NUMBER"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    assert counts == {
        1: 8,
        2: 4,
        3: 2,
        4: 1,
    }
