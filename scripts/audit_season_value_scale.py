from pathlib import Path

import pandas as pd


PATH = Path(
    "data/processed/league_player_season_values.parquet"
)

df = pd.read_parquet(PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

value_columns = [
    column
    for column in df.columns
    if "VALUE" in column.upper()
    and pd.api.types.is_numeric_dtype(df[column])
]

print("\nPossible value columns:")
print(value_columns)

for column in value_columns:
    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        continue

    print("\n" + "=" * 80)
    print(column)
    print("=" * 80)

    print(
        values.quantile(
            [
                0.00,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        ).to_string()
    )

    print("\nHighest values:")

    display_columns = [
        candidate
        for candidate in [
            "PLAYER_NAME",
            "SEASON",
            "SEASON_TYPE",
            column,
        ]
        if candidate in df.columns
    ]

    print(
        df.nlargest(15, column)[display_columns]
        .to_string(index=False)
    )
