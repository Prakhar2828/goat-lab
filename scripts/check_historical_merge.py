import pandas as pd

df = pd.read_parquet(
    "data/processed/goat_player_features.parquet"
)

jordan_regular = df[
    (df["PLAYER_NAME"] == "Michael Jordan")
    & (df["SEASON_TYPE"] == "Regular Season")
].sort_values("SEASON")

jordan_playoffs = df[
    (df["PLAYER_NAME"] == "Michael Jordan")
    & (df["SEASON_TYPE"] == "Playoffs")
].sort_values("SEASON")

lebron_regular = df[
    (df["PLAYER_NAME"] == "LeBron James")
    & (df["SEASON_TYPE"] == "Regular Season")
].sort_values("SEASON")

print("Overall coverage:")
print(df["SEASON"].min(), "through", df["SEASON"].max())

print("\nMichael Jordan regular seasons:")
print(jordan_regular["SEASON"].tolist())
print("Count:", jordan_regular["SEASON"].nunique())

print("\nMichael Jordan playoff seasons:")
print(jordan_playoffs["SEASON"].tolist())
print("Count:", jordan_playoffs["SEASON"].nunique())

print("\nLeBron James regular-season count:")
print(lebron_regular["SEASON"].nunique())

columns = [
    "SEASON",
    "GP",
    "MIN",
    "PTS_PER75",
    "AST_PER75",
    "REB_PER75",
    "TS_PCT",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "PER",
    "WS",
    "BPM",
    "VORP",
    "CAREER_YEAR",
]

columns = [
    column
    for column in columns
    if column in jordan_regular.columns
]

print("\nJordan regular-season metrics:")
print(jordan_regular[columns].to_string(index=False))

duplicates = df.duplicated(
    ["PLAYER_ID", "SEASON", "SEASON_TYPE"],
    keep=False,
)

print("\nDuplicate target rows:", int(duplicates.sum()))
