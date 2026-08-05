from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.playoff_game_evidence import (
    TARGET_PLAYERS,
    PlayoffGamePolicy,
    add_season_relative_metrics,
    bootstrap_series_comparison,
    core_stat_coverage,
    load_playoff_player_games,
    match_candidate_games,
    summarize_player_games,
    summarize_series_games,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
CONFIG_PATH = ROOT / "configs" / "playoff_game_evidence.json"


def write_text_report(
    path: Path,
    metadata: dict[str, object],
    player_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    lines = [
        "GOAT Lab playoff game evidence audit",
        "=" * 40,
        "",
        json.dumps(metadata, indent=2, sort_keys=True),
        "",
        "Player summary",
        "-" * 40,
        player_summary.to_string(index=False),
        "",
        "Series-cluster bootstrap",
        "-" * 40,
        bootstrap.to_string(index=False),
        "",
        "Policy note",
        "-" * 40,
        (
            "Game-level evidence is diagnostic only. It does not change "
            "the central playoff score in this patch."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_registry() -> pd.DataFrame:
    rows = [
        {
            "COMPONENT": "complete_core_box_scores",
            "AVAILABILITY": "available",
            "STATUS": "complete_for_all_481_candidate_games",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "true_shooting_percentage",
            "AVAILABILITY": "available",
            "STATUS": "calculated_from_points_fga_and_fta",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "hollinger_game_score",
            "AVAILABILITY": "available",
            "STATUS": "transparent_box_score_composite",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "season_relative_game_score",
            "AVAILABILITY": "available",
            "STATUS": "standardized_within_same_playoff_season",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "series_cluster_bootstrap",
            "AVAILABILITY": "available",
            "STATUS": "uncertainty_resampled_by_series_not_game",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "elimination_closeout_game_flags",
            "AVAILABILITY": "available",
            "STATUS": "reconstructed_from_pre_game_series_state",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "player_game_plus_minus",
            "AVAILABILITY": "partial",
            "STATUS": "team_dependent_and_not_required_for_core_audit",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
        {
            "COMPONENT": "possession_level_clutch_value",
            "AVAILABILITY": "unavailable",
            "STATUS": "not_inferred_from_box_scores",
            "PRIMARY_MODEL_ELIGIBLE": False,
            "ADDITIONAL_CENTRAL_WEIGHT": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = PlayoffGamePolicy(
        minimum_baseline_minutes=float(config["minimum_baseline_minutes"]),
        bootstrap_repetitions=int(config["bootstrap_repetitions"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        additional_central_weight=float(
            config["central_score_policy"]["additional_central_weight"]
        ),
        final_simulation_allowed=bool(config["final_simulation_allowed"]),
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)

    source_path = (
        DATA
        / "external"
        / "nba_game_history"
        / "PlayerStatistics.csv"
    )
    series_path = PROCESSED / "candidate_series_context_audit.parquet"

    playoff_pool = load_playoff_player_games(source_path)
    playoff_pool = add_season_relative_metrics(
        playoff_pool,
        minimum_minutes=policy.minimum_baseline_minutes,
    )

    candidate_series = pd.read_parquet(series_path)
    candidate_series = candidate_series[
        candidate_series["PLAYER_NAME"].isin(TARGET_PLAYERS)
    ].copy()

    candidate_games = match_candidate_games(
        playoff_pool,
        candidate_series,
    )

    coverage = core_stat_coverage(candidate_games)
    series_summary = summarize_series_games(candidate_games)
    player_summary = summarize_player_games(candidate_games)
    bootstrap = bootstrap_series_comparison(
        series_summary,
        metrics=config["comparison_metrics"],
        repetitions=policy.bootstrap_repetitions,
        seed=policy.bootstrap_seed,
    )
    registry = build_registry()

    expected_games = int(
        pd.to_numeric(
            candidate_series["SERIES_GAMES"],
            errors="coerce",
        ).sum()
    )
    counts = (
        candidate_games.groupby("PLAYER_NAME")["GAME_ID"]
        .nunique()
        .to_dict()
    )
    exact_series = int(
        candidate_games.groupby(["PLAYER_NAME", "SERIES_ID"])["GAME_ID"]
        .nunique()
        .reset_index(name="MATCHED_GAMES")
        .merge(
            candidate_series[
                ["PLAYER_NAME", "SERIES_ID", "SERIES_GAMES"]
            ],
            on=["PLAYER_NAME", "SERIES_ID"],
            how="left",
            validate="one_to_one",
        )
        .eval("MATCHED_GAMES == SERIES_GAMES")
        .sum()
    )
    target_duplicates = int(
        candidate_games.duplicated(
            ["PLAYER_NAME", "GAME_ID"],
            keep=False,
        ).sum()
    )
    minimum_core_coverage = float(coverage["COVERAGE_RATE"].min())

    blockers: list[str] = []
    if expected_games != len(candidate_games):
        blockers.append("candidate_game_total_mismatch")
    if exact_series != len(candidate_series):
        blockers.append("series_game_count_mismatch")
    if target_duplicates:
        blockers.append("duplicate_target_player_games")
    if minimum_core_coverage < 1.0:
        blockers.append("incomplete_core_box_score_coverage")
    if set(player_summary["PLAYER_NAME"]) != set(TARGET_PLAYERS):
        blockers.append("missing_target_player")

    metadata: dict[str, object] = {
        "players": int(player_summary["PLAYER_NAME"].nunique()),
        "candidate_games": len(candidate_games),
        "expected_candidate_games": expected_games,
        "lebron_games": int(counts.get("LeBron James", 0)),
        "jordan_games": int(counts.get("Michael Jordan", 0)),
        "candidate_series": len(candidate_series),
        "exact_game_count_series": exact_series,
        "target_duplicate_player_games": target_duplicates,
        "minimum_core_stat_coverage": minimum_core_coverage,
        "playoff_baseline_games": len(playoff_pool),
        "playoff_baseline_seasons": int(playoff_pool["SEASON"].nunique()),
        "bootstrap_repetitions": policy.bootstrap_repetitions,
        "bootstrap_cluster_unit": "playoff_series",
        "primary_model_eligible": False,
        "additional_central_weight_total": policy.additional_central_weight,
        "central_scores_changed": False,
        "release_blockers": len(blockers),
        "blocker_details": blockers,
        "final_simulation_allowed": policy.final_simulation_allowed,
    }

    candidate_games.to_parquet(
        PROCESSED / "candidate_playoff_games.parquet",
        index=False,
    )
    coverage.to_parquet(
        PROCESSED / "playoff_game_core_coverage.parquet",
        index=False,
    )
    series_summary.to_parquet(
        PROCESSED / "playoff_game_series_summary.parquet",
        index=False,
    )
    player_summary.to_parquet(
        PROCESSED / "playoff_game_player_summary.parquet",
        index=False,
    )
    bootstrap.to_parquet(
        PROCESSED / "playoff_game_bootstrap_comparison.parquet",
        index=False,
    )
    registry.to_parquet(
        PROCESSED / "playoff_game_metric_registry.parquet",
        index=False,
    )
    (PROCESSED / "playoff_game_audit.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_text_report(
        PROCESSED / "playoff_game_audit.txt",
        metadata,
        player_summary,
        bootstrap,
    )

    print("Candidate playoff games:", metadata["candidate_games"])
    print(
        "LeBron/Jordan games:",
        metadata["lebron_games"],
        "/",
        metadata["jordan_games"],
    )
    print(
        "Exact series matches:",
        metadata["exact_game_count_series"],
        "/",
        metadata["candidate_series"],
    )
    print(
        "Minimum core-stat coverage:",
        f"{metadata['minimum_core_stat_coverage']:.3f}",
    )
    print("\nPlayer summary:")
    print(player_summary.to_string(index=False))
    print("\nSeries-cluster bootstrap:")
    print(bootstrap.to_string(index=False))
    print("\nRelease blockers:", metadata["release_blockers"])
    print("Final simulation remains blocked.")

    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
