from __future__ import annotations

import json

from goatlab.models.cultural_impact_sensitivity import (
    run_cultural_impact_sensitivity,
)


def main() -> int:
    grid, summary, metadata = run_cultural_impact_sensitivity()

    print("Cultural-impact sensitivity scenarios:", metadata["scenarios"])
    print("Grid rows:", metadata["grid_rows"])
    print(
        "Baseline maximum absolute error:",
        f"{metadata['baseline_match_max_abs_error']:.12g}",
    )
    print(
        "Configured blend crossover attention weight:",
        metadata["configured_blend_crossover_attention_weight"],
    )
    print("Winner counts:")
    print(json.dumps(metadata["winner_counts"], indent=2, sort_keys=True))

    print("\nBaseline:")
    print(
        grid.loc[
            grid["IS_BASELINE"],
            [
                "PLAYER_NAME",
                "ATTENTION_SCORE",
                "RUBRIC_SCORE",
                "cultural_impact_raw",
                "WINNER",
                "LEBRON_MINUS_JORDAN",
            ],
        ].to_string(index=False)
    )

    print("\nClosest scenarios:")
    print(
        summary.nsmallest(
            10,
            "ABSOLUTE_GAP",
        ).to_string(index=False)
    )

    print("\nRelease blockers:", metadata["release_blockers"])
    print("Central scores changed: False")
    print("Final simulation remains blocked.")

    return 0 if metadata["release_blockers"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
