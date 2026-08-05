"""Run the frozen evidence-uncertainty audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.evidence_uncertainty import (
    build_category_uncertainty,
    build_uncertainty_audit_metadata,
    load_uncertainty_config,
    summarize_expert_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "evidence_uncertainty.json"


def _optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> None:
    config = load_uncertainty_config(CONFIG)

    category_scores = pd.read_parquet(
        PROCESSED / "category_scores.parquet"
    )
    expert = _optional_parquet(
        PROCESSED / "expert_film_consensus.parquet"
    )
    defense = _optional_parquet(
        PROCESSED / "defense_evidence_scores.parquet"
    )

    diagnostics = summarize_expert_diagnostics(
        expert,
        central_score_weight=float(
            config["expert_policy"][
                "central_score_weight"
            ]
        ),
    )

    uncertainty = build_category_uncertainty(
        category_scores,
        config,
        expert_consensus=expert,
        defense_evidence=defense,
    )

    metadata = build_uncertainty_audit_metadata(
        uncertainty,
        diagnostics,
        config,
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)

    uncertainty_path = (
        PROCESSED / "category_uncertainty.parquet"
    )
    diagnostics_path = (
        PROCESSED
        / "expert_diagnostic_summary.parquet"
    )
    json_path = (
        PROCESSED
        / "evidence_uncertainty_audit.json"
    )
    text_path = (
        PROCESSED
        / "evidence_uncertainty_audit.txt"
    )

    uncertainty.to_parquet(
        uncertainty_path,
        index=False,
    )
    diagnostics.to_parquet(
        diagnostics_path,
        index=False,
    )
    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    widest = uncertainty.sort_values(
        "INTERVAL_HALF_WIDTH",
        ascending=False,
    ).head(12)

    lines = [
        "GOAT Lab v1 Evidence Uncertainty Audit",
        "=" * 39,
        "",
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        "",
        "Widest category intervals:",
        widest[
            [
                "PLAYER_NAME",
                "CATEGORY",
                "SCORE",
                "SCORE_LOW",
                "SCORE_HIGH",
                "INTERVAL_HALF_WIDTH",
                "EVIDENCE_STATUS",
            ]
        ].to_string(index=False),
        "",
    ]
    text_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "Expert primary-eligible rows:",
        metadata[
            "expert_primary_eligible_rows"
        ],
    )
    print(
        "Expert used in central score:",
        metadata[
            "expert_used_in_central_score"
        ],
    )
    print(
        "Expert used to narrow intervals:",
        metadata[
            "expert_used_to_narrow_intervals"
        ],
    )
    print()
    print("Widest intervals:")
    print(
        widest[
            [
                "PLAYER_NAME",
                "CATEGORY",
                "INTERVAL_HALF_WIDTH",
            ]
        ].to_string(index=False)
    )
    print()
    print("Wrote", uncertainty_path)
    print("Wrote", diagnostics_path)
    print("Wrote", json_path)
    print("Wrote", text_path)
    print()
    print("Final simulation remains blocked.")


if __name__ == "__main__":
    main()
