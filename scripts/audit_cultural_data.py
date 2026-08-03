from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/wikimedia_pageviews.parquet"
)

TARGET_PLAYERS = {
    "Michael Jordan",
    "LeBron James",
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing expected input: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    required = {
        "PLAYER_NAME",
        "date",
        "views",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["PLAYER_NAME"] = (
        df["PLAYER_NAME"]
        .astype(str)
        .str.strip()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["views"] = pd.to_numeric(
        df["views"],
        errors="coerce",
    )

    duplicates = df.duplicated(
        [
            "PLAYER_NAME",
            "date",
        ],
        keep=False,
    )

    print("=" * 90)
    print("WIKIMEDIA DATA AUDIT")
    print("=" * 90)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nRows by player:")
    print(
        df["PLAYER_NAME"]
        .value_counts()
        .to_string()
    )

    print("\nValidation:")
    print(
        "Duplicate player-date rows:",
        int(duplicates.sum()),
    )
    print(
        "Missing dates:",
        int(df["date"].isna().sum()),
    )
    print(
        "Missing views:",
        int(df["views"].isna().sum()),
    )
    print(
        "Negative views:",
        int((df["views"] < 0).sum()),
    )

    summary = (
        df.groupby(
            "PLAYER_NAME",
            as_index=False,
        )
        .agg(
            START_DATE=("date", "min"),
            END_DATE=("date", "max"),
            DAYS=("date", "nunique"),
            TOTAL_VIEWS=("views", "sum"),
            MEAN_DAILY_VIEWS=("views", "mean"),
            MEDIAN_DAILY_VIEWS=(
                "views",
                "median",
            ),
            P90_DAILY_VIEWS=(
                "views",
                lambda values: values.quantile(
                    0.90
                ),
            ),
            P95_DAILY_VIEWS=(
                "views",
                lambda values: values.quantile(
                    0.95
                ),
            ),
            MAX_DAILY_VIEWS=("views", "max"),
        )
    )

    print("\nOverall coverage:")
    print(
        summary.to_string(
            index=False,
        )
    )

    coverage = summary.set_index(
        "PLAYER_NAME"
    )

    common_start = coverage[
        "START_DATE"
    ].max()

    common_end = coverage[
        "END_DATE"
    ].min()

    if pd.isna(common_start) or pd.isna(
        common_end
    ):
        raise ValueError(
            "Unable to determine a common date window."
        )

    common = df[
        df["date"].between(
            common_start,
            common_end,
        )
    ].copy()

    common_summary = (
        common.groupby(
            "PLAYER_NAME",
            as_index=False,
        )
        .agg(
            DAYS=("date", "nunique"),
            TOTAL_VIEWS=("views", "sum"),
            MEAN_DAILY_VIEWS=("views", "mean"),
            MEDIAN_DAILY_VIEWS=(
                "views",
                "median",
            ),
            P95_DAILY_VIEWS=(
                "views",
                lambda values: values.quantile(
                    0.95
                ),
            ),
        )
    )

    total_common_views = (
        common_summary[
            "TOTAL_VIEWS"
        ].sum()
    )

    common_summary[
        "SHARE_OF_TWO_PLAYER_VIEWS"
    ] = (
        common_summary["TOTAL_VIEWS"]
        / total_common_views
    )

    print(
        "\nCommon comparison window:",
        common_start.date(),
        "through",
        common_end.date(),
    )

    print("\nCommon-window summary:")
    print(
        common_summary.to_string(
            index=False,
        )
    )

    common["YEAR"] = (
        common["date"].dt.year
    )

    yearly = (
        common.groupby(
            [
                "YEAR",
                "PLAYER_NAME",
            ],
            as_index=False,
        )["views"]
        .sum()
    )

    yearly_total = (
        yearly.groupby("YEAR")["views"]
        .transform("sum")
    )

    yearly["VIEW_SHARE"] = (
        yearly["views"]
        / yearly_total
    )

    annual_views = yearly.pivot(
        index="YEAR",
        columns="PLAYER_NAME",
        values="views",
    )

    annual_shares = yearly.pivot(
        index="YEAR",
        columns="PLAYER_NAME",
        values="VIEW_SHARE",
    )

    print("\nAnnual pageviews:")
    print(
        annual_views.to_string()
    )

    print("\nAnnual two-player view shares:")
    print(
        annual_shares.to_string()
    )

    monthly = (
        common.assign(
            MONTH=common["date"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby(
            [
                "MONTH",
                "PLAYER_NAME",
            ],
            as_index=False,
        )["views"]
        .sum()
    )

    top_months = (
        monthly.sort_values(
            "views",
            ascending=False,
        )
        .groupby(
            "PLAYER_NAME",
            as_index=False,
        )
        .head(10)
        .sort_values(
            [
                "PLAYER_NAME",
                "views",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print("\nTen highest-attention months per player:")
    print(
        top_months.to_string(
            index=False,
        )
    )

    assert set(
        df["PLAYER_NAME"].unique()
    ) == TARGET_PLAYERS

    assert not duplicates.any()
    assert df["date"].notna().all()
    assert df["views"].notna().all()
    assert (df["views"] >= 0).all()

    print(
        "\nWikimedia cultural-data audit passed."
    )


if __name__ == "__main__":
    main()
