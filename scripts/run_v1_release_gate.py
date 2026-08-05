from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.category_hierarchy import (
    load_hierarchy_config,
)
from goatlab.models.final_model import (
    score_frozen_hierarchy,
)
from goatlab.models.release_gate import (
    evaluate_v1_release_gate,
)
from goatlab.settings import settings
from goatlab.utils import write_parquet


def main() -> int:
    settings.ensure_directories()

    checks, metadata, production_scores = (
        evaluate_v1_release_gate(
            processed_dir=settings.processed_dir,
        )
    )

    output = Path(settings.processed_dir)
    write_parquet(
        checks,
        output / "v1_release_gate_checks.parquet",
    )
    write_parquet(
        production_scores,
        output / "production_category_scores.parquet",
    )

    hierarchy = load_hierarchy_config(
        "configs/category_hierarchy.json"
    )
    central = score_frozen_hierarchy(
        production_scores,
        hierarchy,
    )
    write_parquet(
        central,
        output / "production_hierarchy_scores.parquet",
    )

    (
        output / "v1_release_gate.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    failed = checks.loc[
        ~checks["PASSED"],
        [
            "CHECK",
            "SEVERITY",
            "VALUE",
            "EXPECTED",
            "RELEASE_BLOCKER",
        ],
    ]

    lines = [
        "GOAT Lab version 1 release gate",
        "================================",
        "",
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        "",
        "Failed or advisory checks:",
        (
            failed.to_string(index=False)
            if not failed.empty
            else "None"
        ),
        "",
        "Frozen central scores:",
        central[
            [
                "PLAYER_NAME",
                "GOAT_SCORE",
                "RANK",
            ]
        ].to_string(index=False),
    ]
    (
        output / "v1_release_gate.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=== RELEASE GATE METADATA ===")
    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    print("\n=== FAILED OR ADVISORY CHECKS ===")
    print(
        failed.to_string(index=False)
        if not failed.empty
        else "None"
    )

    print("\n=== FROZEN CENTRAL SCORES ===")
    print(
        central[
            [
                "PLAYER_NAME",
                "GOAT_SCORE",
                "RANK",
            ]
        ].to_string(index=False)
    )

    if metadata["final_simulation_allowed"]:
        print(
            "\nPatch 10 gate passed. "
            "The final simulation is now eligible "
            "but has not been run."
        )
        return 0

    print(
        "\nPatch 10 gate failed. "
        "Do not run the final simulation."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
