from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.team_context_evidence import (
    attach_context_to_candidate_series,
    build_context_registry,
    build_supporting_cast_context,
    summarize_supporting_cast,
)


def main() -> None:
    root = Path.cwd()
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    season_values_path = processed / "league_player_season_values.parquet"
    series_path = processed / "playoff_series_scored.parquet"

    season_values = pd.read_parquet(season_values_path)
    series_scored = pd.read_parquet(series_path)

    supporting_cast = build_supporting_cast_context(season_values)
    player_summary = summarize_supporting_cast(supporting_cast)
    registry = build_context_registry()
    series_context = attach_context_to_candidate_series(
        series_scored,
        supporting_cast,
    )

    team_strength_columns = [
        "TEAM_SRS",
        "OPP_SRS",
        "TEAM_NET_RATING",
        "OPP_NET_RATING",
    ]
    team_strength_complete = bool(
        all(column in series_context.columns for column in team_strength_columns)
        and series_context[team_strength_columns].notna().all(axis=1).all()
    )

    metadata = {
        "players": int(player_summary["PLAYER_NAME"].nunique()),
        "supporting_cast_rows": len(supporting_cast),
        "support_rows_with_value": int(supporting_cast["SUPPORT_VALUE"].notna().sum()),
        "candidate_series_rows": len(series_context),
        "candidate_series_with_support_context": int(
            series_context["SUPPORT_VALUE"].notna().sum()
        ),
        "injury_context_rows": int(
            supporting_cast["INJURY_CONTEXT_AVAILABLE"].sum()
        ),
        "context_components": len(registry),
        "unavailable_context_components": int(
            registry["AVAILABILITY"].eq("unavailable").sum()
        ),
        "additional_central_weight_total": float(
            registry["ADDITIONAL_CENTRAL_WEIGHT"].sum()
        ),
        "existing_series_model_uses_team_strength": team_strength_complete,
        "support_context_used_in_expectation": False,
        "injury_context_used_in_expectation": False,
        "central_scores_changed": False,
        "release_blockers": 0,
        "final_simulation_allowed": False,
    }

    outputs = {
        "supporting_cast_context.parquet": supporting_cast,
        "supporting_cast_player_summary.parquet": player_summary,
        "team_context_component_registry.parquet": registry,
        "candidate_series_context_audit.parquet": series_context,
    }
    for name, frame in outputs.items():
        path = processed / name
        frame.to_parquet(path, index=False)
        print(f"Wrote {path}")

    json_path = processed / "team_context_audit.json"
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")

    lines = [
        "Team context evidence audit",
        "===========================",
        f"Players: {metadata['players']}",
        f"Supporting-cast rows: {metadata['supporting_cast_rows']}",
        f"Candidate series rows: {metadata['candidate_series_rows']}",
        f"Existing series team-strength context complete: {team_strength_complete}",
        "Supporting-cast context used in expectation: False",
        "Injury context available: False",
        "Central scores changed: False",
        "Final simulation remains blocked.",
        "",
        player_summary.to_string(index=False),
    ]
    text_path = processed / "team_context_audit.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {text_path}")

    print()
    print(player_summary.to_string(index=False))
    print()
    print("Supporting-cast context used in expectation: False")
    print("Injury context available: False")
    print("Central scores changed: False")
    print("Final simulation remains blocked.")


if __name__ == "__main__":
    main()
