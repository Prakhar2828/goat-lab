from __future__ import annotations

import argparse
import json
import subprocess

from goatlab.reporting.final_release import (
    build_release_manifest,
    verify_release_bundle,
    write_release_bundle,
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export or verify the frozen GOAT Lab v1 result bundle."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the committed release bundle and artifact hashes.",
    )
    args = parser.parse_args()

    if args.verify:
        verify_release_bundle()
        print("GOAT Lab v1 release bundle verified.")
        return 0

    manifest = build_release_manifest(
        git_commit=_git_output("rev-parse", "HEAD"),
        git_branch=_git_output("branch", "--show-current"),
    )
    write_release_bundle(manifest)

    print("=== FINAL RESULT ===")
    print(
        json.dumps(
            {
                "classification": manifest["result_classification"],
                "central_result": manifest["central_result"],
                "simulation_result": manifest["simulation_result"],
                "scale_sensitivity": manifest["scale_sensitivity"],
                "cultural_weighting_sensitivity": manifest[
                    "cultural_weighting_sensitivity"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print()
    print("Wrote docs/V1_FINAL_RESULTS.md")
    print("Wrote release/v1_release_manifest.json")
    print("Wrote release/v1_artifact_hashes.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
