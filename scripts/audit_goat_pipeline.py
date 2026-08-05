from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")

TARGET_IDS = {893, 2544}
TARGET_NAMES = {"Michael Jordan", "LeBron James"}


def filter_targets(frame: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=frame.index)

    for column in [
        "PLAYER_ID",
        "player_id",
    ]:
        if column in frame.columns:
            ids = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            mask = mask | ids.isin(TARGET_IDS)

    for column in [
        "PLAYER_NAME",
        "player_name",
        "PLAYER",
        "player",
    ]:
        if column in frame.columns:
            names = (
                frame[column]
                .astype(str)
                .str.strip()
            )
            mask = mask | names.isin(TARGET_NAMES)

    return frame.loc[mask].copy()


def inspect_processed_file(filename: str) -> None:
    path = PROCESSED / filename

    print("\n" + "=" * 100)
    print(filename)
    print("=" * 100)

    if not path.exists():
        print("File does not exist.")
        return

    frame = pd.read_parquet(path)

    print("Full shape:", frame.shape)
    print("Columns:")
    print(frame.columns.tolist())

    if frame.empty:
        print("File is empty.")
        return

    targets = filter_targets(frame)

    if targets.empty:
        print(
            "\nNo Jordan or LeBron rows were identifiable "
            "from the available columns."
        )
        print("\nFirst five rows only:")
        print(frame.head(5).to_string(index=False))
        return

    sort_columns = [
        column
        for column in [
            "PLAYER_NAME",
            "player_name",
            "SEASON",
            "season",
            "SEASON_TYPE",
            "season_type",
        ]
        if column in targets.columns
    ]

    if sort_columns:
        targets = targets.sort_values(sort_columns)

    print("\nJordan and LeBron rows:")
    print(targets.to_string(index=False))


features = pd.read_parquet(
    PROCESSED / "goat_player_features.parquet"
)

print("=" * 100)
print("TARGET FEATURE COVERAGE")
print("=" * 100)

print("Shape:", features.shape)

print("\nRows by player and season type:")
print(
    features.groupby(
        ["PLAYER_NAME", "SEASON_TYPE"]
    ).size()
)

duplicates = features.duplicated(
    [
        "PLAYER_ID",
        "SEASON",
        "SEASON_TYPE",
    ],
    keep=False,
)

print(
    "\nDuplicate target player-season rows:",
    int(duplicates.sum()),
)

jordan_playoffs = features[
    (features["PLAYER_NAME"] == "Michael Jordan")
    & (features["SEASON_TYPE"] == "Playoffs")
].sort_values("SEASON")

lebron_playoffs = features[
    (features["PLAYER_NAME"] == "LeBron James")
    & (features["SEASON_TYPE"] == "Playoffs")
].sort_values("SEASON")

print("\nMichael Jordan playoff games:")
print(
    jordan_playoffs[
        ["SEASON", "GP"]
    ].to_string(index=False)
)

print(
    "Jordan career playoff games:",
    pd.to_numeric(
        jordan_playoffs["GP"],
        errors="coerce",
    ).sum(),
)

print("\nLeBron James playoff games:")
print(
    lebron_playoffs[
        ["SEASON", "GP"]
    ].to_string(index=False)
)

print(
    "LeBron career playoff games:",
    pd.to_numeric(
        lebron_playoffs["GP"],
        errors="coerce",
    ).sum(),
)

core_metrics = [
    column
    for column in [
        "PTS_PER75",
        "AST_PER75",
        "REB_PER75",
        "STL_PER75",
        "BLK_PER75",
        "TS_PCT",
    ]
    if column in features.columns
]

advanced_metrics = [
    column
    for column in [
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "PIE",
        "PER",
        "WS",
        "WS_PER_48",
        "OBPM",
        "DBPM",
        "BPM",
        "VORP",
    ]
    if column in features.columns
]

print("\nCore metric coverage:")
print(
    features.groupby(
        ["PLAYER_NAME", "SEASON_TYPE"]
    )[core_metrics]
    .count()
    .T
)

print("\nAdvanced metric coverage:")
if advanced_metrics:
    print(
        features.groupby(
            ["PLAYER_NAME", "SEASON_TYPE"]
        )[advanced_metrics]
        .count()
        .T
    )
else:
    print("No advanced metrics found.")

print("\nJordan career-year sequence:")
print(
    features[
        features["PLAYER_NAME"] == "Michael Jordan"
    ][
        [
            "SEASON",
            "SEASON_TYPE",
            "CAREER_YEAR",
        ]
    ]
    .sort_values(
        ["SEASON", "SEASON_TYPE"]
    )
    .to_string(index=False)
)

for filename in [
    "goat_player_season_values.parquet",
    "peak_prime_longevity.parquet",
    "category_scores.parquet",
    "historical_career_reference.parquet",
]:
    inspect_processed_file(filename)
