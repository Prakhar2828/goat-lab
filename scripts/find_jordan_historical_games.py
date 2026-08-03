from pathlib import Path

import pandas as pd


ROOT = Path("data/external/nba_game_history")
TARGET_ID = 893


def jordan_mask(frame: pd.DataFrame) -> pd.Series:
    first = (
        frame.get("firstName", pd.Series("", index=frame.index))
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    last = (
        frame.get("lastName", pd.Series("", index=frame.index))
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    by_name = (
        (first == "michael")
        & (last == "jordan")
    )

    if "personId" in frame.columns:
        person_id = pd.to_numeric(
            frame["personId"],
            errors="coerce",
        )

        by_id = person_id == TARGET_ID
    else:
        by_id = pd.Series(False, index=frame.index)

    return by_name | by_id


for filename in [
    "Players.csv",
    "PlayerStatistics.csv",
    "PlayerStatisticsExtended.csv",
]:
    path = ROOT / filename

    print("=" * 100)
    print("FILE:", path)

    if not path.exists():
        print("Missing file")
        continue

    matches = []
    total_rows = 0
    minimum_date = None
    maximum_date = None

    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
        low_memory=False,
    ):
        total_rows += len(chunk)

        date_column = next(
            (
                column
                for column in [
                    "gameDate",
                    "gameDateTimeEst",
                ]
                if column in chunk.columns
            ),
            None,
        )

        if date_column is not None:
            dates = pd.to_datetime(
                chunk[date_column],
                errors="coerce",
            )

            chunk_min = dates.min()
            chunk_max = dates.max()

            if pd.notna(chunk_min):
                minimum_date = (
                    chunk_min
                    if minimum_date is None
                    else min(minimum_date, chunk_min)
                )

            if pd.notna(chunk_max):
                maximum_date = (
                    chunk_max
                    if maximum_date is None
                    else max(maximum_date, chunk_max)
                )

        mask = jordan_mask(chunk)

        if mask.any():
            matches.append(chunk.loc[mask].copy())

    print("Rows scanned:", total_rows)

    if minimum_date is not None:
        print("Date coverage:", minimum_date, "through", maximum_date)

    if not matches:
        print("No Michael Jordan rows found.")
        continue

    jordan = pd.concat(
        matches,
        ignore_index=True,
    )

    print("Michael Jordan rows:", len(jordan))

    if "personId" in jordan.columns:
        print(
            "Person IDs:",
            sorted(
                pd.to_numeric(
                    jordan["personId"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
        )

    if "gameType" in jordan.columns:
        print("\nRows by game type:")
        print(
            jordan["gameType"]
            .fillna("Missing")
            .value_counts()
            .to_string()
        )

    date_column = next(
        (
            column
            for column in [
                "gameDate",
                "gameDateTimeEst",
            ]
            if column in jordan.columns
        ),
        None,
    )

    if date_column is not None:
        jordan["_DATE"] = pd.to_datetime(
            jordan[date_column],
            errors="coerce",
        )

        print(
            "\nJordan date range:",
            jordan["_DATE"].min(),
            "through",
            jordan["_DATE"].max(),
        )

        playoff_rows = jordan[
            jordan.get(
                "gameType",
                pd.Series("", index=jordan.index),
            )
            .astype(str)
            .str.casefold()
            .eq("playoffs")
        ].copy()

        if not playoff_rows.empty:
            playoff_rows["SEASON_END_YEAR"] = (
                playoff_rows["_DATE"].dt.year
            )

            print("\nJordan playoff games by ending year:")
            print(
                playoff_rows.groupby(
                    "SEASON_END_YEAR"
                ).size().to_string()
            )

    display_columns = [
        column
        for column in [
            "firstName",
            "lastName",
            "personId",
            "gameId",
            "gameDate",
            "gameDateTimeEst",
            "gameType",
            "playerteamName",
            "points",
            "assists",
            "reboundsTotal",
            "numMinutes",
        ]
        if column in jordan.columns
    ]

    print("\nFirst Jordan rows:")
    print(
        jordan[display_columns]
        .sort_values(
            date_column
            if date_column is not None
            else display_columns[0]
        )
        .head(15)
        .to_string(index=False)
    )
