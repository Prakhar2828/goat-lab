from __future__ import annotations

import numpy as np
import pandas as pd


def _top_k_mean(
    values: pd.Series,
    k: int,
) -> float:
    clean = (
        pd.to_numeric(values, errors="coerce")
        .dropna()
        .sort_values(ascending=False)
    )

    if clean.empty:
        return float("nan")

    return float(
        clean.head(
            min(k, len(clean))
        ).mean()
    )


def derive_season_value_thresholds(
    player_features: pd.DataFrame,
    value_column: str = "SEASON_VALUE_0_100",
    minimum_minutes: float = 1_000,
    minimum_games: int = 40,
    all_star_quantile: float = 0.75,
    all_nba_quantile: float = 0.90,
    elite_quantile: float = 0.975,
) -> dict[str, float]:
    """Derive thresholds from qualified historical regular seasons."""

    regular = player_features[
        player_features["SEASON_TYPE"]
        == "Regular Season"
    ].copy()

    if value_column not in regular.columns:
        raise ValueError(
            f"Missing season-value column: {value_column}"
        )

    values = pd.to_numeric(
        regular[value_column],
        errors="coerce",
    )

    if "MIN" in regular.columns:
        minutes = pd.to_numeric(
            regular["MIN"],
            errors="coerce",
        )
    else:
        minutes = pd.Series(
            np.inf,
            index=regular.index,
            dtype="float64",
        )

    if "GP" in regular.columns:
        games = pd.to_numeric(
            regular["GP"],
            errors="coerce",
        )
    else:
        games = pd.Series(
            np.inf,
            index=regular.index,
            dtype="float64",
        )

    qualified_mask = (
        values.notna()
        & (minutes >= minimum_minutes)
        & (games >= minimum_games)
    )

    qualified_values = values.loc[
        qualified_mask
    ].dropna()

    # Small fixtures or unusual datasets may contain no qualifying rows.
    # Fall back to all available regular-season values in that situation.
    if qualified_values.empty:
        qualified_values = values.dropna()

    if qualified_values.empty:
        raise ValueError(
            "No regular-season values were available "
            "for threshold calculation."
        )

    return {
        "all_star_threshold": float(
            qualified_values.quantile(
                all_star_quantile
            )
        ),
        "all_nba_threshold": float(
            qualified_values.quantile(
                all_nba_quantile
            )
        ),
        "elite_threshold": float(
            qualified_values.quantile(
                elite_quantile
            )
        ),
    }


def _get_ordering_year(
    frame: pd.DataFrame,
) -> pd.Series:
    """Return an ordering value for consecutive-season calculations."""

    if "SEASON" in frame.columns:
        return pd.to_numeric(
            frame["SEASON"]
            .astype(str)
            .str[:4],
            errors="coerce",
        )

    if "CAREER_YEAR" in frame.columns:
        return pd.to_numeric(
            frame["CAREER_YEAR"],
            errors="coerce",
        )

    return pd.Series(
        np.arange(len(frame)),
        index=frame.index,
        dtype="float64",
    )


def _consecutive_window_max(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
) -> float:
    """Return the best full consecutive-season window.

    Calendar gaps break a window when SEASON is present. For test
    fixtures and other callers without SEASON, CAREER_YEAR is used.
    """

    if value_column not in frame.columns:
        return float("nan")

    ordered = pd.DataFrame(
        {
            "_ORDER_YEAR": _get_ordering_year(
                frame
            ),
            "_VALUE": pd.to_numeric(
                frame[value_column],
                errors="coerce",
            ),
        },
        index=frame.index,
    )

    ordered = ordered.dropna(
        subset=[
            "_ORDER_YEAR",
            "_VALUE",
        ]
    )

    ordered = (
        ordered.sort_values("_ORDER_YEAR")
        .drop_duplicates(
            "_ORDER_YEAR",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(ordered) < window:
        return float("nan")

    year_difference = ordered[
        "_ORDER_YEAR"
    ].diff()

    ordered["_CONSECUTIVE_RUN"] = (
        year_difference.ne(1).cumsum()
    )

    candidates: list[float] = []

    for _, run in ordered.groupby(
        "_CONSECUTIVE_RUN"
    ):
        if len(run) < window:
            continue

        rolling = (
            run["_VALUE"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .dropna()
        )

        if not rolling.empty:
            candidates.append(
                float(rolling.max())
            )

    if not candidates:
        return float("nan")

    return float(max(candidates))


def _count_seasons(
    group: pd.DataFrame,
    value_column: str,
) -> int:
    """Count seasons while supporting compact test fixtures."""

    if "SEASON" in group.columns:
        return int(
            group["SEASON"].nunique()
        )

    if "CAREER_YEAR" in group.columns:
        return int(
            pd.to_numeric(
                group["CAREER_YEAR"],
                errors="coerce",
            ).nunique()
        )

    return int(
        pd.to_numeric(
            group[value_column],
            errors="coerce",
        ).notna().sum()
    )


def summarize_peak_prime_longevity(
    player_features: pd.DataFrame,
    value_column: str = "SEASON_VALUE_0_100",
    all_star_threshold: float = 65.0,
    all_nba_threshold: float = 72.5,
    elite_threshold: float = 80.0,
) -> pd.DataFrame:
    """Summarize peak, prime, and longevity for each player.

    The fixed defaults preserve backward compatibility for direct
    callers and tests. The production category pipeline passes the
    qualified historical thresholds explicitly.
    """

    regular = player_features[
        player_features["SEASON_TYPE"]
        == "Regular Season"
    ].copy()

    if value_column not in regular.columns:
        raise ValueError(
            f"Missing season-value column: {value_column}"
        )

    has_player_id = (
        "PLAYER_ID" in regular.columns
    )

    if has_player_id:
        grouped = regular.groupby(
            [
                "PLAYER_ID",
                "PLAYER_NAME",
            ],
            dropna=False,
        )
    else:
        grouped = regular.groupby(
            "PLAYER_NAME",
            dropna=False,
        )

    rows: list[
        dict[str, float | int | str]
    ] = []

    for group_key, group in grouped:
        if has_player_id:
            player_id, player_name = group_key
        else:
            player_id = None
            player_name = group_key

        values = pd.to_numeric(
            group[value_column],
            errors="coerce",
        )

        row: dict[
            str,
            float | int | str,
        ] = {
            "PLAYER_NAME": str(player_name),
            "BEST_SEASON": float(
                values.max()
            ),
            "TOP_3_PEAK": _top_k_mean(
                values,
                3,
            ),
            "BEST_3_CONSECUTIVE": (
                _consecutive_window_max(
                    group,
                    value_column,
                    3,
                )
            ),
            "BEST_5_CONSECUTIVE": (
                _consecutive_window_max(
                    group,
                    value_column,
                    5,
                )
            ),
            "BEST_7_CONSECUTIVE": (
                _consecutive_window_max(
                    group,
                    value_column,
                    7,
                )
            ),
            "TOP_10_SEASONS": _top_k_mean(
                values,
                10,
            ),
            "ELITE_SEASONS": int(
                (
                    values
                    >= float(elite_threshold)
                ).sum()
            ),
            "ALL_NBA_LEVEL_SEASONS": int(
                (
                    values
                    >= float(all_nba_threshold)
                ).sum()
            ),
            "CAREER_VALUE_ABOVE_AVERAGE": float(
                np.maximum(
                    values - 50.0,
                    0.0,
                ).sum()
            ),
            "CAREER_VALUE_ABOVE_ALL_STAR": float(
                np.maximum(
                    values
                    - float(all_star_threshold),
                    0.0,
                ).sum()
            ),
            "SEASONS": _count_seasons(
                group,
                value_column,
            ),
            "TOTAL_MINUTES": float(
                pd.to_numeric(
                    group.get("MIN", 0),
                    errors="coerce",
                ).sum()
            ),
        }

        if player_id is not None:
            row["PLAYER_ID"] = int(
                player_id
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    preferred_order = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "BEST_SEASON",
        "TOP_3_PEAK",
        "BEST_3_CONSECUTIVE",
        "BEST_5_CONSECUTIVE",
        "BEST_7_CONSECUTIVE",
        "TOP_10_SEASONS",
        "ELITE_SEASONS",
        "ALL_NBA_LEVEL_SEASONS",
        "CAREER_VALUE_ABOVE_AVERAGE",
        "CAREER_VALUE_ABOVE_ALL_STAR",
        "SEASONS",
        "TOTAL_MINUTES",
    ]

    return result[
        [
            column
            for column in preferred_order
            if column in result.columns
        ]
    ]
