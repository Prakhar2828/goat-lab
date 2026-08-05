import pandas as pd

df = pd.read_parquet(
    "data/processed/league_player_season_values.parquet"
)

df["MIN"] = pd.to_numeric(
    df["MIN"],
    errors="coerce",
)

df["GP"] = pd.to_numeric(
    df["GP"],
    errors="coerce",
)

df["SEASON_VALUE_0_100"] = pd.to_numeric(
    df["SEASON_VALUE_0_100"],
    errors="coerce",
)

qualified = df[
    (df["SEASON_TYPE"] == "Regular Season")
    & (df["MIN"] >= 1_000)
    & (df["GP"] >= 40)
    & df["SEASON_VALUE_0_100"].notna()
].copy()

print("Qualified rows:", len(qualified))
print(
    "Coverage:",
    qualified["SEASON"].min(),
    "through",
    qualified["SEASON"].max(),
)

quantiles = qualified["SEASON_VALUE_0_100"].quantile(
    [
        0.50,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.975,
        0.99,
    ]
)

print("\nQualified regular-season quantiles:")
print(quantiles.to_string())

all_nba_threshold = float(
    qualified["SEASON_VALUE_0_100"].quantile(0.90)
)

elite_threshold = float(
    qualified["SEASON_VALUE_0_100"].quantile(0.975)
)

print("\nSuggested thresholds:")
print(
    "All-NBA-level:",
    round(all_nba_threshold, 4),
)
print(
    "Elite:",
    round(elite_threshold, 4),
)

targets = qualified[
    qualified["PLAYER_NAME"].isin(
        ["Michael Jordan", "LeBron James"]
    )
].copy()

targets["ALL_NBA_LEVEL"] = (
    targets["SEASON_VALUE_0_100"]
    >= all_nba_threshold
)

targets["ELITE"] = (
    targets["SEASON_VALUE_0_100"]
    >= elite_threshold
)

print("\nJordan and LeBron qualified seasons:")
print(
    targets[
        [
            "PLAYER_NAME",
            "SEASON",
            "GP",
            "MIN",
            "SEASON_VALUE_0_100",
            "ALL_NBA_LEVEL",
            "ELITE",
        ]
    ]
    .sort_values(
        [
            "PLAYER_NAME",
            "SEASON",
        ]
    )
    .to_string(index=False)
)

print("\nCounts:")
print(
    targets.groupby("PLAYER_NAME")[
        [
            "ALL_NBA_LEVEL",
            "ELITE",
        ]
    ].sum()
)
