from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.impact_evidence import write_impact_audit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"

    metadata = write_impact_audit(
        processed / "goat_player_season_values.parquet",
        processed,
    )

    coverage = pd.read_parquet(
        processed / "impact_metric_coverage.parquet"
    )
    registry = pd.read_parquet(
        processed / "impact_metric_registry.parquet"
    )

    print("Impact metric policy:")
    print(
        registry[
            [
                "METRIC",
                "AVAILABILITY",
                "STATUS",
                "PRIMARY_MODEL_ELIGIBLE",
                "CENTRAL_SCORE_WEIGHT",
            ]
        ].to_string(index=False)
    )

    print("\nCoverage:")
    print(
        coverage[
            [
                "PLAYER_NAME",
                "METRIC",
                "OBSERVATIONS",
                "TOTAL_ROWS",
                "COVERAGE_RATE",
                "STATUS",
            ]
        ].to_string(index=False)
    )

    print("\nAudit metadata:")
    print(json.dumps(metadata, indent=2, sort_keys=True))

    print("\nWrote data/processed/impact_metric_values.parquet")
    print("Wrote data/processed/impact_metric_coverage.parquet")
    print("Wrote data/processed/impact_metric_registry.parquet")
    print("Wrote data/processed/impact_metric_audit.json")
    print("\nCentral category scores remain unchanged.")
    print("Final simulation remains blocked.")


if __name__ == "__main__":
    main()
