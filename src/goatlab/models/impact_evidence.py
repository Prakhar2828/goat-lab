from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET_PLAYERS = ("Michael Jordan", "LeBron James")

UNAVAILABLE_METRICS = (
    "on_off_net",
    "rapm",
    "epm",
    "lebron_metric",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    candidates: tuple[str, ...]
    status: str
    reason: str


METRIC_SPECS = (
    MetricSpec(
        "net_rating",
        ("NET_RATING_y", "NET_RATING", "NET_RATING_x"),
        "diagnostic_only_provenance_mismatch",
        "Available values are not verified true on/off splits and may differ in construction across eras.",
    ),
    MetricSpec(
        "pie",
        ("PIE_y", "PIE", "PIE_x"),
        "diagnostic_only_coverage_asymmetry",
        "PIE coverage is materially different across the two careers.",
    ),
    MetricSpec(
        "plus_minus",
        ("PLUS_MINUS",),
        "diagnostic_only_coverage_asymmetry",
        "Raw plus-minus is unavailable for most of Jordan's career and is not adjusted impact.",
    ),
    MetricSpec(
        "ws_per_48",
        ("WS_PER_48_y", "WS_PER_48", "WS_PER_48_x"),
        "diagnostic_only_merge_gap",
        "Merged coverage is incomplete and not comparable across both players.",
    ),
    MetricSpec(
        "bpm",
        ("BPM_y", "BPM", "BPM_x"),
        "diagnostic_only_merge_gap",
        "Merged coverage is incomplete and not comparable across both players.",
    ),
    MetricSpec(
        "vorp",
        ("VORP_y", "VORP", "VORP_x"),
        "diagnostic_only_merge_gap",
        "Merged coverage is incomplete and not comparable across both players.",
    ),
)


def _first_available_value(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    present = [column for column in candidates if column in frame.columns]
    if not present:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")

    values = frame[present].apply(pd.to_numeric, errors="coerce")
    return values.bfill(axis=1).iloc[:, 0].astype("Float64")


def build_impact_metric_values(
    season_values: pd.DataFrame,
    players: tuple[str, ...] = TARGET_PLAYERS,
) -> pd.DataFrame:
    required = {"PLAYER_NAME", "SEASON", "SEASON_TYPE"}
    missing = required.difference(season_values.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = season_values[
        season_values["PLAYER_NAME"].isin(players)
    ].copy()

    rows: list[pd.DataFrame] = []
    for spec in METRIC_SPECS:
        metric = frame[["PLAYER_NAME", "SEASON", "SEASON_TYPE"]].copy()
        metric["METRIC"] = spec.name
        metric["VALUE"] = _first_available_value(frame, spec.candidates)
        metric["STATUS"] = spec.status
        metric["PRIMARY_MODEL_ELIGIBLE"] = False
        metric["CENTRAL_SCORE_WEIGHT"] = 0.0
        rows.append(metric)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_impact_coverage(
    values: pd.DataFrame,
    players: tuple[str, ...] = TARGET_PLAYERS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for metric in [spec.name for spec in METRIC_SPECS]:
        metric_rows = values[values["METRIC"].eq(metric)]
        status = next(spec.status for spec in METRIC_SPECS if spec.name == metric)

        for player in players:
            player_rows = metric_rows[
                metric_rows["PLAYER_NAME"].eq(player)
            ]
            observed = int(player_rows["VALUE"].notna().sum())
            total = len(player_rows)
            regular = int(
                player_rows.loc[
                    player_rows["SEASON_TYPE"].eq("Regular Season"),
                    "VALUE",
                ].notna().sum()
            )
            playoffs = int(
                player_rows.loc[
                    player_rows["SEASON_TYPE"].eq("Playoffs"),
                    "VALUE",
                ].notna().sum()
            )
            rows.append(
                {
                    "PLAYER_NAME": player,
                    "METRIC": metric,
                    "OBSERVATIONS": observed,
                    "TOTAL_ROWS": total,
                    "COVERAGE_RATE": observed / total if total else 0.0,
                    "REGULAR_OBSERVATIONS": regular,
                    "PLAYOFF_OBSERVATIONS": playoffs,
                    "STATUS": status,
                    "PRIMARY_MODEL_ELIGIBLE": False,
                    "CENTRAL_SCORE_WEIGHT": 0.0,
                }
            )

    return pd.DataFrame(rows)


def build_metric_registry(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in METRIC_SPECS:
        subset = coverage[coverage["METRIC"].eq(spec.name)]
        rates = subset["COVERAGE_RATE"].astype(float)
        rows.append(
            {
                "METRIC": spec.name,
                "AVAILABILITY": "local_partial",
                "STATUS": spec.status,
                "MIN_PLAYER_COVERAGE": float(rates.min()) if not rates.empty else 0.0,
                "MAX_PLAYER_COVERAGE": float(rates.max()) if not rates.empty else 0.0,
                "PRIMARY_MODEL_ELIGIBLE": False,
                "CENTRAL_SCORE_WEIGHT": 0.0,
                "REASON": spec.reason,
            }
        )

    for metric in UNAVAILABLE_METRICS:
        rows.append(
            {
                "METRIC": metric,
                "AVAILABILITY": "unavailable",
                "STATUS": "unavailable_not_estimated",
                "MIN_PLAYER_COVERAGE": 0.0,
                "MAX_PLAYER_COVERAGE": 0.0,
                "PRIMARY_MODEL_ELIGIBLE": False,
                "CENTRAL_SCORE_WEIGHT": 0.0,
                "REASON": "No reproducible local source is available; the metric is not imputed or estimated.",
            }
        )

    return pd.DataFrame(rows)


def build_impact_audit(
    season_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    values = build_impact_metric_values(season_values)
    coverage = build_impact_coverage(values)
    registry = build_metric_registry(coverage)

    team_impact_rows = (
        int(season_values["FAMILY_TEAM_IMPACT"].notna().sum())
        if "FAMILY_TEAM_IMPACT" in season_values.columns
        else 0
    )
    team_impact_positive_coverage = (
        int(
            pd.to_numeric(
                season_values["COVERAGE_TEAM_IMPACT"],
                errors="coerce",
            ).fillna(0).gt(0).sum()
        )
        if "COVERAGE_TEAM_IMPACT" in season_values.columns
        else 0
    )

    metadata = {
        "players": int(coverage["PLAYER_NAME"].nunique()),
        "metrics_audited": len(registry),
        "local_partial_metrics": int(registry["AVAILABILITY"].eq("local_partial").sum()),
        "unavailable_metrics": int(registry["AVAILABILITY"].eq("unavailable").sum()),
        "primary_eligible_metrics": int(registry["PRIMARY_MODEL_ELIGIBLE"].sum()),
        "central_score_weight_total": float(registry["CENTRAL_SCORE_WEIGHT"].sum()),
        "team_impact_non_null_rows": team_impact_rows,
        "team_impact_positive_coverage_rows": team_impact_positive_coverage,
        "team_impact_used_in_model": bool(team_impact_rows > 0),
        "central_scores_changed": False,
        "release_blockers": 0,
        "final_simulation_allowed": False,
    }
    return values, coverage, registry, metadata


def write_impact_audit(
    season_values_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    season_values = pd.read_parquet(season_values_path)
    values, coverage, registry, metadata = build_impact_audit(season_values)

    output_dir.mkdir(parents=True, exist_ok=True)
    values.to_parquet(output_dir / "impact_metric_values.parquet", index=False)
    coverage.to_parquet(output_dir / "impact_metric_coverage.parquet", index=False)
    registry.to_parquet(output_dir / "impact_metric_registry.parquet", index=False)

    import json

    (output_dir / "impact_metric_audit.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
