from __future__ import annotations

from pathlib import Path

from goatlab.reporting.public_release import verify_public_release_bundle


def test_public_release_bundle_matches_frozen_manifest() -> None:
    result = verify_public_release_bundle(Path("."))
    assert result["winner"] == "LeBron James"
    assert len(result["verified_artifacts"]) == 7
    assert len(result["supporting_dashboard_files"]) == 3
