from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.category_scaling import (
    REFERENCE_CATEGORIES,
    SCALING_SCENARIOS,
    build_saturation_audit,
    build_scaling_comparison,
)
from goatlab.settings import settings
from goatlab.utils import write_parquet


def main() -> None:
    category_path = (
        settings.processed_dir
        / "category_scores.parquet"
    )
    reference_path = (
        settings.processed_dir
        / "historical_career_reference.parquet"
    )

    if not category_path.exists():
        raise FileNotFoundError(
            "Missing category scores. Run "
            "`goatlab build-features` first."
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            "Missing historical career reference. "
            "Run `goatlab build-features` first."
        )

    category_scores = pd.read_parquet(
        category_path
    )
    historical_reference = pd.read_parquet(
        reference_path
    )

    comparison = build_scaling_comparison(
        category_scores,
        historical_reference,
    )
    audit = build_saturation_audit(
        comparison
    )

    comparison_path = (
        settings.processed_dir
        / "category_scaling_comparison.parquet"
    )
    audit_path = (
        settings.processed_dir
        / "category_scaling_saturation_audit.parquet"
    )
    metadata_path = (
        settings.processed_dir
        / "category_scaling_audit.json"
    )
    report_path = (
        settings.processed_dir
        / "category_scaling_audit.txt"
    )

    write_parquet(
        comparison,
        comparison_path,
    )
    write_parquet(
        audit,
        audit_path,
    )

    reference_audit = audit[
        audit["CATEGORY"].isin(
            REFERENCE_CATEGORIES
        )
    ].copy()

    scenario_summary = (
        reference_audit.groupby(
            "SCENARIO",
            as_index=False,
        )
        .agg(
            CATEGORIES=(
                "CATEGORY",
                "nunique",
            ),
            SATURATED_CATEGORIES=(
                "SATURATION_FLAG",
                "sum",
            ),
            MEDIAN_PAIR_GAP=(
                "SCORE_RANGE",
                "median",
            ),
            MIN_PAIR_GAP=(
                "SCORE_RANGE",
                "min",
            ),
            MAX_PAIR_GAP=(
                "SCORE_RANGE",
                "max",
            ),
        )
        .sort_values(
            "SCENARIO"
        )
        .reset_index(
            drop=True
        )
    )

    historical_saturated = int(
        scenario_summary.loc[
            scenario_summary[
                "SCENARIO"
            ].eq(
                "historical_percentile"
            ),
            "SATURATED_CATEGORIES",
        ].iloc[0]
    )

    metadata = {
        "scenarios": list(
            SCALING_SCENARIOS
        ),
        "players": sorted(
            comparison[
                "PLAYER_NAME"
            ].unique().tolist()
        ),
        "reference_categories": list(
            REFERENCE_CATEGORIES
        ),
        "comparison_rows": int(
            len(comparison)
        ),
        "audit_rows": int(
            len(audit)
        ),
        "historical_percentile_saturated_categories": (
            historical_saturated
        ),
        "production_scale_frozen": False,
        "final_simulation_allowed": False,
        "note": (
            "This patch audits scale behavior. "
            "It does not select a production scale."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "GOAT Lab category-scaling audit",
        "================================",
        "",
        scenario_summary.to_string(
            index=False
        ),
        "",
        (
            "Production scale frozen: no"
        ),
        (
            "Final simulation allowed: no"
        ),
        "",
        (
            "Historical percentile saturation "
            "is measured, not silently repaired."
        ),
    ]

    report_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        scenario_summary.to_string(
            index=False
        )
    )
    print()
    print(
        f"Wrote {comparison_path}"
    )
    print(
        f"Wrote {audit_path}"
    )
    print(
        f"Wrote {metadata_path}"
    )
    print(
        f"Wrote {report_path}"
    )
    print()
    print(
        "Final simulation remains blocked."
    )


if __name__ == "__main__":
    main()
