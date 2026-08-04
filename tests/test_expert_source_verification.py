from __future__ import annotations

import json

import pandas as pd
import pytest

from goatlab.data.expert_source_verification import (
    FetchResult,
    build_source_verification_blockers,
    build_verification_record,
    extract_metadata,
    validate_verified_claim_sources,
)


def _source() -> dict[str, object]:
    return {
        "SOURCE_ID": "SOURCE_1",
        "TITLE": (
            "Film Study: Example Analysis"
        ),
        "ANALYST": "Jane Analyst",
        "PUBLICATION": "Example",
        "PUBLICATION_DATE": "2020-05-13",
        "URL": (
            "https://example.com/analysis"
        ),
        "PLAYER_COVERAGE": "LeBron James",
    }


def _html_result(
    title: str = (
        "Film Study: Example Analysis"
    ),
) -> FetchResult:
    html = f"""
    <html>
      <head>
        <meta
          property="og:title"
          content="{title}"
        />
        <meta
          name="author"
          content="Jane Analyst"
        />
        <meta
          property="og:site_name"
          content="Example"
        />
        <meta
          property="article:published_time"
          content="2020-05-13T10:00:00Z"
        />
      </head>
      <body>
        <h1>{title}</h1>
        <p>
          This film analysis breaks down LeBron James,
          transition offense, passing, scoring, shooting,
          defense, and playoff matchup decisions in detail.
          The article includes enough analytical text for
          the content check to identify its purpose.
        </p>
      </body>
    </html>
    """.encode()

    return FetchResult(
        fetch_method="html",
        fetch_url=(
            "https://example.com/analysis"
        ),
        final_url=(
            "https://example.com/analysis"
        ),
        fetch_status="fetched",
        http_status=200,
        content_type="text/html",
        payload=html,
    )


def test_html_metadata_extraction() -> None:
    metadata = extract_metadata(
        _html_result()
    )

    assert metadata[
        "title"
    ] == (
        "Film Study: Example Analysis"
    )

    assert metadata[
        "author"
    ] == "Jane Analyst"

    assert metadata[
        "publication"
    ] == "Example"

    assert metadata[
        "date"
    ].startswith(
        "2020-05-13"
    )


def test_complete_source_passes_automated_checks() -> None:
    record = (
        build_verification_record(
            _source(),
            _html_result(),
            checked_at=(
                "2026-08-04T00:00:00+00:00"
            ),
        )
    )

    assert (
        record[
            "AUTOMATED_STATUS"
        ]
        == "verified_automated"
    )

    assert (
        record[
            "TITLE_MATCH"
        ]
        is True
    )

    assert (
        record[
            "ANALYST_MATCH"
        ]
        is True
    )

    assert (
        record[
            "DATE_MATCH"
        ]
        is True
    )

    assert (
        record[
            "PLAYER_COVERAGE_CHECK"
        ]
        is True
    )


def test_title_mismatch_is_detected() -> None:
    record = (
        build_verification_record(
            _source(),
            _html_result(
                title=(
                    "A Completely Different Article"
                )
            ),
        )
    )

    assert (
        record[
            "AUTOMATED_STATUS"
        ]
        == "title_mismatch"
    )


def test_youtube_oembed_has_metadata_gap() -> None:
    payload = json.dumps(
        {
            "title": (
                "Attention to Detail: "
                "LeBron James"
            ),
            "author_name": (
                "By Any Means Basketball"
            ),
            "provider_name": "YouTube",
        }
    ).encode()

    source = _source()

    source[
        "TITLE"
    ] = (
        "Attention to Detail: "
        "LeBron James"
    )

    source[
        "ANALYST"
    ] = (
        "By Any Means Basketball"
    )

    result = FetchResult(
        fetch_method="youtube_oembed",
        fetch_url=(
            "https://www.youtube.com/oembed"
        ),
        final_url=(
            "https://www.youtube.com/oembed"
        ),
        fetch_status="fetched",
        http_status=200,
        content_type="application/json",
        payload=payload,
    )

    record = (
        build_verification_record(
            source,
            result,
        )
    )

    assert (
        record[
            "AUTOMATED_STATUS"
        ]
        == "verified_with_metadata_gap"
    )

    assert (
        record[
            "ANALYTICAL_CONTENT_CHECK"
        ]
        == ""
    )


def test_pending_review_creates_blocker() -> None:
    sources = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
            }
        ]
    )

    verifications = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
                "AUTOMATED_STATUS": (
                    "verified_automated"
                ),
                "REVIEW_STATUS": "pending",
            }
        ]
    )

    # Add remaining schema fields for the validator.
    from goatlab.data.expert_source_verification import (
        VERIFICATION_COLUMNS,
    )

    for column in VERIFICATION_COLUMNS:
        if column not in verifications:
            verifications[
                column
            ] = ""

    blockers = (
        build_source_verification_blockers(
            sources,
            verifications,
        )
    )

    assert (
        "pending_source_review"
        in set(
            blockers[
                "BLOCKER_TYPE"
            ]
        )
    )


def test_verified_claim_requires_verified_source() -> None:
    claims = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
                "REVIEW_STATUS": "verified",
            }
        ]
    )

    verifications = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
                "REVIEW_STATUS": "pending",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="not passed source review",
    ):
        validate_verified_claim_sources(
            claims,
            verifications,
        )


def _complete_verification_row(
    automated_status: str,
    review_status: str,
    review_notes: str = "",
) -> dict[str, object]:
    from goatlab.data.expert_source_verification import (
        VERIFICATION_COLUMNS,
    )

    row: dict[str, object] = {
        column: ""
        for column in VERIFICATION_COLUMNS
    }

    row.update(
        {
            "SOURCE_ID": "SOURCE_1",
            "AUTOMATED_STATUS": (
                automated_status
            ),
            "REVIEW_STATUS": (
                review_status
            ),
            "REVIEWED_BY": (
                "GOAT Lab source review"
            ),
            "REVIEWED_AT": (
                "2026-08-04T12:00:00+00:00"
            ),
            "REVIEW_NOTES": (
                review_notes
            ),
        }
    )

    return row


def test_documented_qualified_review_can_override_fetch_failure() -> None:
    sources = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
            }
        ]
    )

    verifications = pd.DataFrame(
        [
            _complete_verification_row(
                automated_status=(
                    "fetch_failed"
                ),
                review_status=(
                    "verified_with_qualification"
                ),
                review_notes=(
                    "Official publisher page was "
                    "independently confirmed; the "
                    "automated client received 403."
                ),
            )
        ]
    )

    blockers = (
        build_source_verification_blockers(
            sources,
            verifications,
        )
    )

    assert blockers.empty


def test_qualified_override_requires_review_notes() -> None:
    sources = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
            }
        ]
    )

    verifications = pd.DataFrame(
        [
            _complete_verification_row(
                automated_status=(
                    "fetch_failed"
                ),
                review_status=(
                    "verified_with_qualification"
                ),
                review_notes="",
            )
        ]
    )

    blockers = (
        build_source_verification_blockers(
            sources,
            verifications,
        )
    )

    blocker_types = set(
        blockers[
            "BLOCKER_TYPE"
        ]
    )

    assert (
        "automated_source_verification_failed"
        in blocker_types
    )

    assert (
        "incomplete_qualified_source_review"
        in blocker_types
    )


def test_plain_verified_status_cannot_override_failure() -> None:
    sources = pd.DataFrame(
        [
            {
                "SOURCE_ID": "SOURCE_1",
            }
        ]
    )

    verifications = pd.DataFrame(
        [
            _complete_verification_row(
                automated_status=(
                    "analytical_content_unconfirmed"
                ),
                review_status="verified",
                review_notes=(
                    "Attempted unqualified override."
                ),
            )
        ]
    )

    blockers = (
        build_source_verification_blockers(
            sources,
            verifications,
        )
    )

    assert (
        "invalid_unqualified_source_override"
        in set(
            blockers[
                "BLOCKER_TYPE"
            ]
        )
    )
