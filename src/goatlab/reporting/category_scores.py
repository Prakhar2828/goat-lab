from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from goatlab.models.peak_prime import (
    derive_season_value_thresholds,
    summarize_peak_prime_longevity,
)
from goatlab.models.season_value import add_transparent_season_value
from goatlab.settings import settings
from goatlab.utils import load_yaml, read_optional_parquet, write_parquet


CATEGORIES = [
    "peak",
    "prime",
    "longevity",
    "regular_season",
    "playoffs",
    "winning_context",
    "offense",
    "defense",
    "cultural_impact",
]


def _weighted_group_mean(group: pd.DataFrame, columns: list[str]) -> float:
    available = [column for column in columns if column in group.columns]
    if not available:
        return float("nan")

    values = group[available].mean(axis=1, skipna=True)

    if "MIN" in group.columns:
        weights = pd.to_numeric(group["MIN"], errors="coerce").fillna(0)
    else:
        weights = pd.Series(1.0, index=group.index, dtype="float64")

    valid = values.notna() & (weights > 0)
    if valid.any():
        return float(np.average(values[valid], weights=weights[valid]))

    return float(values.mean())


def _to_reference_percentile(value: float, reference: pd.Series) -> float:
    clean = pd.to_numeric(reference, errors="coerce").dropna()
    if pd.isna(value) or clean.empty:
        return float("nan")

    return float(percentileofscore(clean, value, kind="rank"))


def _prepare_league_reference(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    league = add_transparent_season_value(features.copy())

    first_season = league.groupby("PLAYER_ID")["SEASON"].transform("min")
    season_start_year = pd.to_numeric(
        league["SEASON"].astype(str).str[:4],
        errors="coerce",
    )
    career_start_year = pd.to_numeric(
        first_season.astype(str).str[:4],
        errors="coerce",
    )

    league["CAREER_YEAR"] = (
        season_start_year - career_start_year + 1
    ).astype("Int64")

    thresholds = derive_season_value_thresholds(league)

    regular = league[league["SEASON_TYPE"] == "Regular Season"].copy()
    playoffs = league[league["SEASON_TYPE"] == "Playoffs"].copy()

    peak_summary = summarize_peak_prime_longevity(
        league,
        all_star_threshold=thresholds["all_star_threshold"],
        all_nba_threshold=thresholds["all_nba_threshold"],
        elite_threshold=thresholds["elite_threshold"],
    )

    rows: list[dict[str, float | str | int]] = []

    for (player_id, player_name), regular_group in regular.groupby(
        ["PLAYER_ID", "PLAYER_NAME"],
        dropna=False,
    ):
        if pd.isna(player_id):
            continue

        playoff_group = playoffs[playoffs["PLAYER_ID"] == player_id]
        peak_row = peak_summary[
            peak_summary["PLAYER_ID"] == int(player_id)
        ]

        if peak_row.empty:
            continue

        peak_row = peak_row.iloc[0]
        top_regular = (
            pd.to_numeric(
                regular_group["SEASON_VALUE_0_100"],
                errors="coerce",
            )
            .dropna()
            .sort_values(ascending=False)
            .head(10)
        )

        rows.append(
            {
                "PLAYER_ID": int(player_id),
                "PLAYER_NAME": str(player_name),
                "peak_raw": float(peak_row["TOP_3_PEAK"]),
                "prime_raw": float(peak_row["BEST_7_CONSECUTIVE"]),
                "longevity_raw": float(
                    peak_row["CAREER_VALUE_ABOVE_AVERAGE"]
                ),
                "regular_season_raw": (
                    float(top_regular.mean())
                    if not top_regular.empty
                    else np.nan
                ),
                "playoffs_raw": _weighted_group_mean(
                    playoff_group,
                    ["SEASON_VALUE_0_100"],
                ),
                "offense_raw": _weighted_group_mean(
                    regular_group,
                    ["FAMILY_SCORING", "FAMILY_PLAYMAKING"],
                ),
                "defense_raw": _weighted_group_mean(
                    regular_group,
                    ["FAMILY_DEFENSE_BOX"],
                ),
                "SEASONS": int(regular_group["SEASON"].nunique()),
                "TOTAL_MINUTES": float(
                    pd.to_numeric(
                        regular_group.get("MIN", 0),
                        errors="coerce",
                    ).sum()
                ),
            }
        )

    career_reference = pd.DataFrame(rows)
    if career_reference.empty:
        return league, career_reference, thresholds

    eligible = career_reference[
        (career_reference["SEASONS"] >= 5)
        & (career_reference["TOTAL_MINUTES"] >= 5_000)
    ].copy()

    return league, eligible, thresholds


def _load_context_scores(target: pd.DataFrame) -> pd.DataFrame:
    result = target.copy()
    result["winning_context_raw"] = np.nan
    result["cultural_impact_raw"] = np.nan

    series = read_optional_parquet(
        settings.processed_dir / "playoff_series_scored.parquet"
    )
    required_series_columns = {
        "PLAYER_NAME",
        "SERIES_OVERPERFORMANCE",
    }

    if (
        not series.empty
        and required_series_columns.issubset(series.columns)
    ):
        context = (
            series.dropna(subset=["PLAYER_NAME"])
            .groupby("PLAYER_NAME", as_index=False)["SERIES_OVERPERFORMANCE"]
            .mean()
        )

        # Convert team outcome residual [-1, 1] to a display scale centered on 50.
        context["winning_context_raw"] = (
            50 + 50 * context["SERIES_OVERPERFORMANCE"]
        ).clip(0, 100)

        result = result.merge(
            context[["PLAYER_NAME", "winning_context_raw"]],
            on="PLAYER_NAME",
            how="left",
            suffixes=("", "_model"),
        )
        result["winning_context_raw"] = result[
            "winning_context_raw_model"
        ].combine_first(result["winning_context_raw"])
        result = result.drop(columns=["winning_context_raw_model"])

    manual_path = settings.manual_dir / "manual_category_inputs.csv"
    if manual_path.exists():
        manual = pd.read_csv(manual_path)
        result = result.merge(
            manual,
            on="PLAYER_NAME",
            how="left",
            suffixes=("", "_manual"),
        )

        for column in ["winning_context_raw", "cultural_impact_raw"]:
            manual_column = f"{column}_manual"
            if manual_column in result.columns:
                result[column] = result[manual_column].combine_first(
                    result[column]
                )

        result = result.drop(
            columns=[
                column
                for column in result.columns
                if column.endswith("_manual")
                and column
                not in {
                    "defense_film_score_manual",
                    "defense_awards_score_manual",
                }
            ],
            errors="ignore",
        )
        result = result.rename(
            columns={
                "defense_film_score_manual": "defense_film_score",
                "defense_awards_score_manual": "defense_awards_score",
            }
        )

    return result


def build_category_scores() -> pd.DataFrame:
    features = read_optional_parquet(
        settings.processed_dir / "league_player_features.parquet"
    )
    if features.empty:
        raise FileNotFoundError("Build player features first.")

    league_values, reference, thresholds = _prepare_league_reference(features)

    if reference.empty:
        raise ValueError(
            "No eligible historical career reference players were produced."
        )

    write_parquet(
        league_values,
        settings.processed_dir / "league_player_season_values.parquet",
    )

    source_config = load_yaml("configs/sources.yaml")
    target_ids = {
        int(player["player_id"])
        for player in source_config["players"].values()
    }

    target = reference[reference["PLAYER_ID"].isin(target_ids)].copy()
    missing_target_ids = target_ids.difference(
        set(target["PLAYER_ID"].astype(int))
    )
    if missing_target_ids:
        raise ValueError(
            "Target players are missing from the eligible career reference: "
            f"{sorted(missing_target_ids)}"
        )

    target = _load_context_scores(target)

    reference_columns = {
        "peak": "peak_raw",
        "prime": "prime_raw",
        "longevity": "longevity_raw",
        "regular_season": "regular_season_raw",
        "playoffs": "playoffs_raw",
        "offense": "offense_raw",
        "defense": "defense_raw",
    }

    for category, raw_column in reference_columns.items():
        target[category] = target[raw_column].apply(
            lambda value: _to_reference_percentile(
                value,
                reference[raw_column],
            )
        )

    # Defensive box evidence is combined with optional structured film and awards
    # evidence. Available components are reweighted rather than treating missing
    # evidence as zero.
    def combine_defense(row: pd.Series) -> float:
        components = [
            (row.get("defense"), 0.50),
            (row.get("defense_film_score"), 0.35),
            (row.get("defense_awards_score"), 0.15),
        ]
        available = [
            (float(value), weight)
            for value, weight in components
            if pd.notna(value)
        ]

        if not available:
            return float("nan")

        total_weight = sum(weight for _, weight in available)
        return sum(
            value * weight
            for value, weight in available
        ) / total_weight

    target["defense"] = target.apply(combine_defense, axis=1)

    # These categories use their own 0-100 evidence scales because no complete
    # historical league reference population exists.
    target["winning_context"] = pd.to_numeric(
        target["winning_context_raw"],
        errors="coerce",
    )
    target["cultural_impact"] = pd.to_numeric(
        target["cultural_impact_raw"],
        errors="coerce",
    )

    category = target[["PLAYER_NAME", *CATEGORIES]].sort_values(
        "PLAYER_NAME"
    )
    write_parquet(
        category,
        settings.processed_dir / "category_scores.parquet",
    )

    target_values = league_values[
        league_values["PLAYER_ID"].isin(target_ids)
    ].copy()
    write_parquet(
        target_values,
        settings.processed_dir / "goat_player_season_values.parquet",
    )

    target_summary = summarize_peak_prime_longevity(
        league_values,
        all_star_threshold=thresholds["all_star_threshold"],
        all_nba_threshold=thresholds["all_nba_threshold"],
        elite_threshold=thresholds["elite_threshold"],
    )
    target_summary = target_summary[
        target_summary["PLAYER_ID"].isin(target_ids)
    ].copy()

    write_parquet(
        target_summary,
        settings.processed_dir / "peak_prime_longevity.parquet",
    )
    write_parquet(
        reference,
        settings.processed_dir / "historical_career_reference.parquet",
    )

    return category