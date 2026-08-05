from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goatlab.models.defense_evidence import (
    build_defense_evidence_scores,
)
from goatlab.reporting.category_scores import (
    build_category_scores,
)
from goatlab.settings import settings
from goatlab.utils import write_parquet


def main() -> None:
    settings.ensure_directories()

    awards_path = (
        settings.interim_dir
        / "player_awards.parquet"
    )
    consensus_path = (
        settings.processed_dir
        / "expert_film_consensus.parquet"
    )

    if not awards_path.exists():
        raise FileNotFoundError(
            "Missing player awards data: "
            f"{awards_path}"
        )

    if not consensus_path.exists():
        raise FileNotFoundError(
            "Missing expert film consensus: "
            f"{consensus_path}"
        )

    awards = pd.read_parquet(
        awards_path
    )
    consensus = pd.read_parquet(
        consensus_path
    )

    scores, normalized_awards = (
        build_defense_evidence_scores(
            awards,
            consensus,
        )
    )

    scores_path = (
        settings.processed_dir
        / "defense_evidence_scores.parquet"
    )
    awards_output_path = (
        settings.processed_dir
        / "defensive_awards_normalized.parquet"
    )

    write_parquet(
        scores,
        scores_path,
    )
    write_parquet(
        normalized_awards,
        awards_output_path,
    )

    category_scores = build_category_scores()

    metadata = {
        "players": int(
            scores["PLAYER_NAME"].nunique()
        ),
        "recognized_defensive_award_rows": int(
            len(normalized_awards)
        ),
        "awards_players_covered": int(
            scores[
                "AWARDS_USED_IN_MODEL"
            ].sum()
        ),
        "film_primary_eligible_rows": int(
            scores[
                "DEFENSE_FILM_PRIMARY_ROWS"
            ].sum()
        ),
        "film_used_in_model_players": int(
            scores[
                "FILM_USED_IN_MODEL"
            ].sum()
        ),
        "awards_used_in_model": True,
        "film_used_in_model": bool(
            scores[
                "FILM_USED_IN_MODEL"
            ].any()
        ),
        "release_blockers": int(
            scores[
                "DEFENSE_RELEASE_BLOCKER"
            ].sum()
        ),
        "final_simulation_allowed": False,
    }

    audit_json = (
        settings.processed_dir
        / "defense_evidence_audit.json"
    )
    audit_json.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "GOAT Lab defense evidence audit",
        "",
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        "",
        "Defense evidence scores",
        scores.to_string(
            index=False
        ),
        "",
        "Updated category scores",
        category_scores.to_string(
            index=False
        ),
        "",
        (
            "Expert film remains diagnostic unless "
            "PRIMARY_MODEL_ELIGIBLE is true."
        ),
        "Final simulation remains blocked.",
    ]

    audit_text = (
        settings.processed_dir
        / "defense_evidence_audit.txt"
    )
    audit_text.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8",
    )

    print(
        scores[
            [
                "PLAYER_NAME",
                "DPOY",
                "ALL_DEFENSIVE_FIRST",
                "ALL_DEFENSIVE_SECOND",
                "DEFENSIVE_AWARD_POINTS",
                "defense_awards_score",
                "DEFENSE_FILM_PRIMARY_ROWS",
                "FILM_USED_IN_MODEL",
                "DEFENSE_EVIDENCE_STATUS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Wrote {scores_path}"
    )
    print(
        f"Wrote {awards_output_path}"
    )
    print(
        f"Wrote {audit_json}"
    )
    print(
        f"Wrote {audit_text}"
    )
    print()
    print(
        "Final simulation remains blocked."
    )


if __name__ == "__main__":
    main()
