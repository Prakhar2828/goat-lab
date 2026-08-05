from pathlib import Path

import pandas as pd

ROOT = Path("data/external/nba_game_history")

files = sorted(
    path
    for path in ROOT.rglob("*.csv")
    if ".complete" not in path.parts
)

print(f"Found {len(files)} CSV files\n")

for path in files:
    print("=" * 100)
    print("FILE:", path)

    try:
        frame = pd.read_csv(
            path,
            nrows=5,
            low_memory=False,
        )

        print("COLUMN COUNT:", len(frame.columns))
        print("COLUMNS:")
        print(frame.columns.tolist())

        print("\nSAMPLE:")
        print(frame.head(2).to_string(index=False))

    except Exception as exc:
        print(
            f"FAILED: {type(exc).__name__}: {exc}"
        )

    print()
