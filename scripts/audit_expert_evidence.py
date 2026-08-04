from __future__ import annotations

import argparse
import json

from goatlab.models.expert_evidence import (
    build_expert_consensus,
    build_expert_release_blockers,
    read_expert_evidence,
    score_expert_sources,
    validate_expert_evidence,
)
from goatlab.settings import settings
from goatlab.utils import write_parquet


LINE = "=" * 100


def heading(
    title: str,
) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit unsuccessfully when expert "
            "evidence release blockers remain."
        ),
    )

    args = parser.parse_args()

    settings.ensure_directories()

    (
        sources,
        claims,
        dimensions,
    ) = read_expert_evidence(
        settings.manual_dir
    )

    validate_expert_evidence(
        sources,
        claims,
        dimensions,
    )

    source_quality = (
        score_expert_sources(
            sources
        )
    )

    consensus = (
        build_expert_consensus(
            sources,
            claims,
            dimensions,
        )
    )

    blockers = (
        build_expert_release_blockers(
            sources,
            claims,
            dimensions,
            consensus,
        )
    )

    write_parquet(
        source_quality,
        settings.processed_dir
        / "expert_source_quality.parquet",
    )

    write_parquet(
        consensus,
        settings.processed_dir
        / "expert_film_consensus.parquet",
    )

    write_parquet(
        blockers,
        settings.processed_dir
        / "expert_evidence_blockers.parquet",
    )

    metadata = {
        "registered_sources": int(
            len(sources)
        ),
        "registered_source_families": int(
            sources[
                "SOURCE_FAMILY"
            ].nunique()
            if not sources.empty
            else 0
        ),
        "registered_claims": int(
            len(claims)
        ),
        "verified_claims": int(
            claims[
                "REVIEW_STATUS"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin(
                [
                    "verified",
                    (
                        "verified_with_"
                        "qualification"
                    ),
                ]
            )
            .sum()
            if not claims.empty
            else 0
        ),
        "consensus_rows": int(
            len(consensus)
        ),
        "primary_eligible_rows": int(
            consensus[
                "PRIMARY_MODEL_ELIGIBLE"
            ].sum()
            if not consensus.empty
            else 0
        ),
        "release_blockers": int(
            len(blockers)
        ),
        "release_ready": bool(
            blockers.empty
        ),
    }

    metadata_path = (
        settings.processed_dir
        / "expert_evidence_audit.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    heading(
        "EXPERT SOURCE QUALITY"
    )

    if source_quality.empty:
        print(
            "No sources registered."
        )
    else:
        print(
            source_quality[
                [
                    "SOURCE_ID",
                    "SOURCE_FAMILY",
                    "ANALYST",
                    "PUBLICATION",
                    "SOURCE_QUALITY_SCORE",
                    "SOURCE_TIER",
                ]
            ].to_string(
                index=False
            )
        )

    heading(
        "EXPERT FILM CONSENSUS"
    )

    if consensus.empty:
        print(
            "No verified consensus rows."
        )
    else:
        print(
            consensus.to_string(
                index=False
            )
        )

    heading(
        "RELEASE BLOCKERS"
    )

    if blockers.empty:
        print(
            "No expert-evidence blockers."
        )
    else:
        print(
            blockers.to_string(
                index=False
            )
        )

    heading(
        "AUDIT SUMMARY"
    )

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\nWrote:"
    )

    for path in [
        settings.processed_dir
        / "expert_source_quality.parquet",
        settings.processed_dir
        / "expert_film_consensus.parquet",
        settings.processed_dir
        / "expert_evidence_blockers.parquet",
        metadata_path,
    ]:
        print(path)

    if (
        args.strict
        and not blockers.empty
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
