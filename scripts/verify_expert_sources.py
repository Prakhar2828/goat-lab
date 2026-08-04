from __future__ import annotations

import argparse

from goatlab.data.expert_source_verification import (
    build_source_verification_blockers,
    validate_source_verifications,
    verify_source_registry,
)
from goatlab.models.expert_evidence import (
    read_expert_evidence,
    validate_expert_evidence,
)
from goatlab.settings import settings


LINE = "=" * 100


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    args = parser.parse_args()

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

    output_path = (
        settings.manual_dir
        / "expert_source_verifications.csv"
    )

    verifications = (
        verify_source_registry(
            sources,
            output_path,
            timeout_seconds=args.timeout,
        )
    )

    validate_source_verifications(
        sources,
        verifications,
    )

    blockers = (
        build_source_verification_blockers(
            sources,
            verifications,
        )
    )

    print(LINE)
    print("EXPERT SOURCE VERIFICATION")
    print(LINE)

    print(
        verifications[
            [
                "SOURCE_ID",
                "FETCH_METHOD",
                "HTTP_STATUS",
                "OBSERVED_TITLE",
                "TITLE_MATCH",
                "ANALYST_MATCH",
                "DATE_MATCH",
                "PLAYER_COVERAGE_CHECK",
                "ANALYTICAL_CONTENT_CHECK",
                "AUTOMATED_STATUS",
                "REVIEW_STATUS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(LINE)
    print("AUTOMATED STATUS COUNTS")
    print(LINE)

    print(
        verifications[
            "AUTOMATED_STATUS"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print(LINE)
    print("SOURCE VERIFICATION BLOCKERS")
    print(LINE)

    if blockers.empty:
        print(
            "No source verification blockers."
        )
    else:
        print(
            blockers.to_string(
                index=False
            )
        )

    print()
    print(
        f"Wrote {output_path}"
    )

    if (
        args.strict
        and not blockers.empty
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
