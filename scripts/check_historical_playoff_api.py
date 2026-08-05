from nba_api.stats.endpoints import leaguedashplayerstats

from goatlab.data.nba_client import (
    NBA_STATS_HEADERS,
    patch_nba_api_headers,
)

patch_nba_api_headers()


def check(
    measure_type: str,
    per_mode: str,
) -> None:
    print("=" * 72)
    print(
        f"1984-85 | Playoffs | "
        f"{measure_type} | {per_mode}"
    )

    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season="1984-85",
        season_type_all_star="Playoffs",
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode,
        timeout=120,
        headers=NBA_STATS_HEADERS,
    )

    frame = endpoint.get_data_frames()[0]

    print("Shape:", frame.shape)
    print("Columns:", frame.columns.tolist())

    if not frame.empty:
        jordan = frame[
            frame["PLAYER_NAME"]
            .astype(str)
            .str.contains(
                "Michael Jordan",
                case=False,
                na=False,
            )
        ]

        print("\nMichael Jordan:")
        print(jordan.to_string(index=False))

    print()


requests = [
    ("Base", "Totals"),
    ("Base", "Per100Possessions"),
    ("Advanced", "Totals"),
]

for measure_type, per_mode in requests:
    try:
        check(measure_type, per_mode)
    except Exception as exc:
        print(
            f"FAILED: {measure_type} / {per_mode}\n"
            f"{type(exc).__name__}: {exc}\n"
        )
