from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

PUBLIC_ARTIFACT_MAP = {
    "data/processed/production_category_scores.parquet":
        "release/dashboard_data/production_category_scores.parquet",
    "data/processed/production_hierarchy_scores.parquet":
        "release/dashboard_data/production_hierarchy_scores.parquet",
    "data/processed/weight_simulation_summary.parquet":
        "release/dashboard_data/weight_simulation_summary.parquet",
    "data/processed/weight_simulation_drivers.parquet":
        "release/dashboard_data/weight_simulation_drivers.parquet",
    "data/processed/hierarchy_weight_simulation_group_audit.parquet":
        "release/dashboard_data/hierarchy_weight_simulation_group_audit.parquet",
    "data/processed/v1_release_gate.json":
        "release/dashboard_data/v1_release_gate.json",
    "models/training_metadata.json":
        "release/v1_artifacts/models/training_metadata.json",
}

SUPPORTING_DASHBOARD_FILES = (
    "release/dashboard_data/goat_player_season_values.parquet",
    "release/dashboard_data/peak_prime_longevity.parquet",
    "release/dashboard_data/playoff_series_scored.parquet",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_public_release_bundle(root: Path = Path(".")) -> dict[str, Any]:
    manifest_path = root / "release/v1_release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["artifact_sha256"]

    verified: dict[str, str] = {}
    for original, packaged in PUBLIC_ARTIFACT_MAP.items():
        packaged_path = root / packaged
        if not packaged_path.exists():
            raise FileNotFoundError(packaged_path)

        actual = _sha256(packaged_path)
        expected = str(expected_hashes[original])
        if actual != expected:
            raise ValueError(
                f"Public artifact hash mismatch for {packaged}: "
                f"actual={actual}, expected={expected}"
            )
        verified[packaged] = actual

    for relative_path in SUPPORTING_DASHBOARD_FILES:
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(path)

    scores = pd.read_parquet(
        root / "release/dashboard_data/production_hierarchy_scores.parquet"
    )
    simulation = pd.read_parquet(
        root / "release/dashboard_data/weight_simulation_summary.parquet"
    )

    players = set(scores["PLAYER_NAME"])
    if players != {"LeBron James", "Michael Jordan"}:
        raise ValueError(f"Unexpected production player set: {sorted(players)}")

    if abs(float(simulation["WIN_RATE"].sum()) - 1.0) > 1e-12:
        raise ValueError("Simulation win rates do not sum to one.")

    if manifest["result_classification"] != (
        "conditional_not_robust_across_approved_scaling_scenarios"
    ):
        raise ValueError("Unexpected release classification.")

    return {
        "verified_artifacts": verified,
        "supporting_dashboard_files": list(SUPPORTING_DASHBOARD_FILES),
        "winner": manifest["central_result"]["winner"],
        "source_commit": manifest["source_commit"],
    }
