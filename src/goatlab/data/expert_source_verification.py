from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd


VERIFICATION_COLUMNS = [
    "SOURCE_ID",
    "CHECKED_AT",
    "FETCH_METHOD",
    "FETCH_URL",
    "FETCH_FINAL_URL",
    "FETCH_STATUS",
    "HTTP_STATUS",
    "CONTENT_TYPE",
    "OBSERVED_TITLE",
    "OBSERVED_ANALYST",
    "OBSERVED_PUBLICATION",
    "OBSERVED_PUBLICATION_DATE",
    "TITLE_MATCH",
    "ANALYST_MATCH",
    "DATE_MATCH",
    "PLAYER_COVERAGE_CHECK",
    "ANALYTICAL_CONTENT_CHECK",
    "CONTENT_SHA256",
    "CONTENT_BYTES",
    "AUTOMATED_STATUS",
    "FETCH_ERROR",
    "REVIEW_STATUS",
    "REVIEWED_BY",
    "REVIEWED_AT",
    "REVIEW_NOTES",
]

ACCEPTED_AUTOMATED_STATUSES = {
    "verified_automated",
    "verified_with_metadata_gap",
}

ACCEPTED_REVIEW_STATUSES = {
    "verified",
    "verified_with_qualification",
}

ANALYTICAL_TERMS = {
    "analysis",
    "breakdown",
    "defense",
    "defensive",
    "film",
    "matchup",
    "offense",
    "offensive",
    "passing",
    "playmaking",
    "scoring",
    "shot",
    "shooting",
    "transition",
}


@dataclass(frozen=True)
class FetchResult:
    fetch_method: str
    fetch_url: str
    final_url: str
    fetch_status: str
    http_status: int | None
    content_type: str
    payload: bytes
    error: str = ""


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(
            convert_charrefs=True
        )

        self.in_title = False
        self.in_json_ld = False
        self.ignored_depth = 0

        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.json_ld_parts: list[str] = []

        self.meta: dict[str, str] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[
            tuple[str, str | None]
        ],
    ) -> None:
        lowered = tag.casefold()

        attributes = {
            str(key).casefold(): (
                ""
                if value is None
                else str(value)
            )
            for key, value in attrs
        }

        if lowered == "title":
            self.in_title = True

        if lowered in {
            "script",
            "style",
            "noscript",
            "svg",
        }:
            self.ignored_depth += 1

        if (
            lowered == "script"
            and attributes.get(
                "type",
                "",
            ).casefold()
            == "application/ld+json"
        ):
            self.in_json_ld = True

        if lowered == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).strip().casefold()

            content = attributes.get(
                "content",
                "",
            ).strip()

            if key and content:
                self.meta.setdefault(
                    key,
                    content,
                )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        lowered = tag.casefold()

        if lowered == "title":
            self.in_title = False

        if lowered == "script":
            self.in_json_ld = False

        if (
            lowered
            in {
                "script",
                "style",
                "noscript",
                "svg",
            }
            and self.ignored_depth > 0
        ):
            self.ignored_depth -= 1

    def handle_data(
        self,
        data: str,
    ) -> None:
        cleaned = data.strip()

        if not cleaned:
            return

        if self.in_title:
            self.title_parts.append(
                cleaned
            )

        if self.in_json_ld:
            self.json_ld_parts.append(
                cleaned
            )

        if self.ignored_depth == 0:
            self.text_parts.append(
                cleaned
            )


def _normalize_text(
    value: object,
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value or ""),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    text = unescape(
        text
    ).casefold()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _clean_text(
    value: object,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        unescape(
            str(value or "")
        ),
    ).strip()


def _similarity(
    expected: object,
    observed: object,
) -> float:
    expected_normalized = (
        _normalize_text(
            expected
        )
    )

    observed_normalized = (
        _normalize_text(
            observed
        )
    )

    if (
        not expected_normalized
        or not observed_normalized
    ):
        return 0.0

    if (
        expected_normalized
        in observed_normalized
        or observed_normalized
        in expected_normalized
    ):
        return 1.0

    return float(
        SequenceMatcher(
            None,
            expected_normalized,
            observed_normalized,
        ).ratio()
    )


def _person_match(
    expected: object,
    observed: object,
) -> bool | None:
    expected_normalized = (
        _normalize_text(
            expected
        )
    )

    observed_normalized = (
        _normalize_text(
            observed
        )
    )

    if not observed_normalized:
        return None

    if (
        expected_normalized
        in observed_normalized
        or observed_normalized
        in expected_normalized
    ):
        return True

    expected_tokens = (
        expected_normalized.split()
    )

    if not expected_tokens:
        return None

    surname = expected_tokens[-1]

    return surname in set(
        observed_normalized.split()
    )


def _date_prefix(
    value: object,
) -> str:
    match = re.search(
        r"\d{4}-\d{2}-\d{2}",
        str(value or ""),
    )

    return (
        match.group(0)
        if match
        else ""
    )


def _date_match(
    expected: object,
    observed: object,
) -> bool | None:
    observed_date = _date_prefix(
        observed
    )

    if not observed_date:
        return None

    return (
        _date_prefix(
            expected
        )
        == observed_date
    )


def _is_youtube_url(
    url: str,
) -> bool:
    hostname = (
        urlparse(
            url
        )
        .hostname
        or ""
    ).casefold()

    return hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }


def _youtube_oembed_url(
    source_url: str,
) -> str:
    return (
        "https://www.youtube.com/oembed?"
        + urlencode(
            {
                "url": source_url,
                "format": "json",
            }
        )
    )


def fetch_source(
    source_url: str,
    timeout_seconds: int = 30,
    max_bytes: int = 5_000_000,
) -> FetchResult:
    if _is_youtube_url(
        source_url
    ):
        fetch_method = "youtube_oembed"
        fetch_url = _youtube_oembed_url(
            source_url
        )
    else:
        fetch_method = "html"
        fetch_url = source_url

    request = Request(
        fetch_url,
        headers={
            "User-Agent": (
                "GOAT-Lab/0.1 "
                "source-verification "
                "(https://github.com/"
                "Prakhar2828/goat-lab)"
            ),
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/json"
            ),
            "Accept-Encoding": "identity",
        },
        method="GET",
    )

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            payload = response.read(
                max_bytes + 1
            )

            if len(payload) > max_bytes:
                raise ValueError(
                    "Source response exceeded "
                    f"{max_bytes} bytes."
                )

            content_type = (
                response.headers.get(
                    "Content-Type",
                    "",
                )
                .split(
                    ";",
                    1,
                )[0]
                .strip()
                .casefold()
            )

            return FetchResult(
                fetch_method=fetch_method,
                fetch_url=fetch_url,
                final_url=response.geturl(),
                fetch_status="fetched",
                http_status=int(
                    response.status
                ),
                content_type=content_type,
                payload=payload,
            )

    except HTTPError as exc:
        return FetchResult(
            fetch_method=fetch_method,
            fetch_url=fetch_url,
            final_url=str(
                exc.geturl()
            ),
            fetch_status="http_error",
            http_status=int(
                exc.code
            ),
            content_type="",
            payload=b"",
            error=str(
                exc
            ),
        )

    except (
        URLError,
        TimeoutError,
        ValueError,
        OSError,
    ) as exc:
        return FetchResult(
            fetch_method=fetch_method,
            fetch_url=fetch_url,
            final_url="",
            fetch_status="fetch_error",
            http_status=None,
            content_type="",
            payload=b"",
            error=repr(
                exc
            ),
        )


def _iter_json_nodes(
    value: Any,
) -> list[dict[str, Any]]:
    nodes: list[
        dict[str, Any]
    ] = []

    if isinstance(
        value,
        dict,
    ):
        nodes.append(
            value
        )

        for nested in value.values():
            nodes.extend(
                _iter_json_nodes(
                    nested
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for nested in value:
            nodes.extend(
                _iter_json_nodes(
                    nested
                )
            )

    return nodes


def _json_ld_metadata(
    parts: list[str],
) -> dict[str, str]:
    output: dict[
        str,
        str,
    ] = {}

    for part in parts:
        try:
            parsed = json.loads(
                part
            )
        except json.JSONDecodeError:
            continue

        for node in _iter_json_nodes(
            parsed
        ):
            if (
                "title"
                not in output
            ):
                value = (
                    node.get("headline")
                    or node.get("name")
                )

                if isinstance(
                    value,
                    str,
                ):
                    output[
                        "title"
                    ] = _clean_text(
                        value
                    )

            if (
                "date"
                not in output
            ):
                value = (
                    node.get(
                        "datePublished"
                    )
                    or node.get(
                        "uploadDate"
                    )
                )

                if isinstance(
                    value,
                    str,
                ):
                    output[
                        "date"
                    ] = _clean_text(
                        value
                    )

            if (
                "author"
                not in output
            ):
                author = node.get(
                    "author"
                )

                if isinstance(
                    author,
                    str,
                ):
                    output[
                        "author"
                    ] = _clean_text(
                        author
                    )

                elif isinstance(
                    author,
                    dict,
                ):
                    value = author.get(
                        "name"
                    )

                    if isinstance(
                        value,
                        str,
                    ):
                        output[
                            "author"
                        ] = _clean_text(
                            value
                        )

                elif isinstance(
                    author,
                    list,
                ):
                    names = [
                        item.get(
                            "name"
                        )
                        for item in author
                        if isinstance(
                            item,
                            dict,
                        )
                        and isinstance(
                            item.get(
                                "name"
                            ),
                            str,
                        )
                    ]

                    if names:
                        output[
                            "author"
                        ] = ", ".join(
                            names
                        )

            publisher = node.get(
                "publisher"
            )

            if (
                "publication"
                not in output
                and isinstance(
                    publisher,
                    dict,
                )
                and isinstance(
                    publisher.get(
                        "name"
                    ),
                    str,
                )
            ):
                output[
                    "publication"
                ] = _clean_text(
                    publisher[
                        "name"
                    ]
                )

    return output


def extract_metadata(
    result: FetchResult,
) -> dict[str, str]:
    if not result.payload:
        return {
            "title": "",
            "author": "",
            "publication": "",
            "date": "",
            "body": "",
        }

    if (
        result.fetch_method
        == "youtube_oembed"
        or result.content_type
        == "application/json"
    ):
        try:
            payload = json.loads(
                result.payload.decode(
                    "utf-8"
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return {
                "title": "",
                "author": "",
                "publication": "",
                "date": "",
                "body": "",
            }

        title = _clean_text(
            payload.get(
                "title",
                "",
            )
        )

        author = _clean_text(
            payload.get(
                "author_name",
                "",
            )
        )

        publication = _clean_text(
            payload.get(
                "provider_name",
                "",
            )
        )

        return {
            "title": title,
            "author": author,
            "publication": publication,
            "date": "",
            "body": " ".join(
                value
                for value in [
                    title,
                    author,
                    publication,
                ]
                if value
            ),
        }

    text = result.payload.decode(
        "utf-8",
        errors="replace",
    )

    parser = _MetadataParser()

    try:
        parser.feed(
            text
        )
    except Exception:
        pass

    json_ld = _json_ld_metadata(
        parser.json_ld_parts
    )

    title = _clean_text(
        parser.meta.get(
            "og:title",
            "",
        )
        or parser.meta.get(
            "twitter:title",
            "",
        )
        or json_ld.get(
            "title",
            "",
        )
        or " ".join(
            parser.title_parts
        )
    )

    author = _clean_text(
        parser.meta.get(
            "author",
            "",
        )
        or parser.meta.get(
            "article:author",
            "",
        )
        or json_ld.get(
            "author",
            "",
        )
    )

    publication = _clean_text(
        parser.meta.get(
            "og:site_name",
            "",
        )
        or json_ld.get(
            "publication",
            "",
        )
    )

    date = _clean_text(
        parser.meta.get(
            "article:published_time",
            "",
        )
        or parser.meta.get(
            "date",
            "",
        )
        or parser.meta.get(
            "datepublished",
            "",
        )
        or json_ld.get(
            "date",
            "",
        )
    )

    body = _clean_text(
        " ".join(
            parser.text_parts
        )
    )

    return {
        "title": title,
        "author": author,
        "publication": publication,
        "date": date,
        "body": body,
    }


def _player_coverage_check(
    expected_coverage: object,
    searchable_text: str,
) -> bool | None:
    coverage = _normalize_text(
        expected_coverage
    )

    searchable = _normalize_text(
        searchable_text
    )

    if not searchable:
        return None

    jordan_present = (
        "michael jordan"
        in searchable
    )

    lebron_present = (
        "lebron james"
        in searchable
    )

    if coverage == "both":
        return (
            jordan_present
            and lebron_present
        )

    if coverage == "michael jordan":
        return jordan_present

    if coverage == "lebron james":
        return lebron_present

    return None


def _analytical_content_check(
    body: str,
    fetch_method: str,
) -> bool | None:
    normalized = _normalize_text(
        body
    )

    if fetch_method == "youtube_oembed":
        # oEmbed verifies identity and availability,
        # but it does not expose the full analysis.
        return None

    if len(normalized) < 200:
        return False

    tokens = set(
        normalized.split()
    )

    return bool(
        tokens
        & ANALYTICAL_TERMS
    )


def _boolean_or_blank(
    value: bool | None,
) -> bool | str:
    return (
        ""
        if value is None
        else bool(value)
    )


def build_verification_record(
    source: Mapping[str, object],
    result: FetchResult,
    checked_at: str | None = None,
) -> dict[str, object]:
    metadata = extract_metadata(
        result
    )

    expected_title = source.get(
        "TITLE",
        "",
    )

    observed_title = metadata[
        "title"
    ]

    title_match: bool | None

    if observed_title:
        title_match = (
            _similarity(
                expected_title,
                observed_title,
            )
            >= 0.82
        )
    else:
        title_match = None

    analyst_match = _person_match(
        source.get(
            "ANALYST",
            "",
        ),
        metadata[
            "author"
        ],
    )

    date_match = _date_match(
        source.get(
            "PUBLICATION_DATE",
            "",
        ),
        metadata[
            "date"
        ],
    )

    searchable = " ".join(
        [
            metadata[
                "title"
            ],
            metadata[
                "body"
            ],
        ]
    )

    player_coverage_check = (
        _player_coverage_check(
            source.get(
                "PLAYER_COVERAGE",
                "",
            ),
            searchable,
        )
    )

    analytical_content_check = (
        _analytical_content_check(
            metadata[
                "body"
            ],
            result.fetch_method,
        )
    )

    if (
        result.fetch_status
        != "fetched"
    ):
        automated_status = (
            "fetch_failed"
        )

    elif title_match is False:
        automated_status = (
            "title_mismatch"
        )

    elif analyst_match is False:
        automated_status = (
            "analyst_mismatch"
        )

    elif date_match is False:
        automated_status = (
            "publication_date_mismatch"
        )

    elif (
        player_coverage_check
        is False
    ):
        automated_status = (
            "player_coverage_mismatch"
        )

    elif (
        analytical_content_check
        is False
    ):
        automated_status = (
            "analytical_content_unconfirmed"
        )

    elif any(
        value is None
        for value in [
            title_match,
            analyst_match,
            date_match,
            player_coverage_check,
            analytical_content_check,
        ]
    ):
        automated_status = (
            "verified_with_metadata_gap"
        )

    else:
        automated_status = (
            "verified_automated"
        )

    return {
        "SOURCE_ID": str(
            source.get(
                "SOURCE_ID",
                "",
            )
        ),
        "CHECKED_AT": (
            checked_at
            or datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "FETCH_METHOD": (
            result.fetch_method
        ),
        "FETCH_URL": (
            result.fetch_url
        ),
        "FETCH_FINAL_URL": (
            result.final_url
        ),
        "FETCH_STATUS": (
            result.fetch_status
        ),
        "HTTP_STATUS": (
            ""
            if result.http_status is None
            else result.http_status
        ),
        "CONTENT_TYPE": (
            result.content_type
        ),
        "OBSERVED_TITLE": (
            metadata["title"]
        ),
        "OBSERVED_ANALYST": (
            metadata["author"]
        ),
        "OBSERVED_PUBLICATION": (
            metadata["publication"]
        ),
        "OBSERVED_PUBLICATION_DATE": (
            _date_prefix(
                metadata["date"]
            )
        ),
        "TITLE_MATCH": (
            _boolean_or_blank(
                title_match
            )
        ),
        "ANALYST_MATCH": (
            _boolean_or_blank(
                analyst_match
            )
        ),
        "DATE_MATCH": (
            _boolean_or_blank(
                date_match
            )
        ),
        "PLAYER_COVERAGE_CHECK": (
            _boolean_or_blank(
                player_coverage_check
            )
        ),
        "ANALYTICAL_CONTENT_CHECK": (
            _boolean_or_blank(
                analytical_content_check
            )
        ),
        "CONTENT_SHA256": (
            hashlib.sha256(
                result.payload
            ).hexdigest()
            if result.payload
            else ""
        ),
        "CONTENT_BYTES": len(
            result.payload
        ),
        "AUTOMATED_STATUS": (
            automated_status
        ),
        "FETCH_ERROR": (
            result.error
        ),
        "REVIEW_STATUS": "pending",
        "REVIEWED_BY": "",
        "REVIEWED_AT": "",
        "REVIEW_NOTES": "",
    }


def _previous_reviews(
    path: Path,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    existing = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    if (
        "SOURCE_ID"
        not in existing.columns
    ):
        return {}

    return {
        str(row["SOURCE_ID"]): {
            "REVIEW_STATUS": str(
                row.get(
                    "REVIEW_STATUS",
                    "",
                )
            ),
            "REVIEWED_BY": str(
                row.get(
                    "REVIEWED_BY",
                    "",
                )
            ),
            "REVIEWED_AT": str(
                row.get(
                    "REVIEWED_AT",
                    "",
                )
            ),
            "REVIEW_NOTES": str(
                row.get(
                    "REVIEW_NOTES",
                    "",
                )
            ),
        }
        for row in existing.to_dict(
            orient="records"
        )
    }


def verify_source_registry(
    sources: pd.DataFrame,
    output_path: Path,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    previous = _previous_reviews(
        output_path
    )

    records: list[
        dict[str, object]
    ] = []

    for source in sources.to_dict(
        orient="records"
    ):
        source_id = str(
            source["SOURCE_ID"]
        )

        result = fetch_source(
            str(
                source["URL"]
            ),
            timeout_seconds=(
                timeout_seconds
            ),
        )

        record = (
            build_verification_record(
                source,
                result,
            )
        )

        if source_id in previous:
            for field in [
                "REVIEW_STATUS",
                "REVIEWED_BY",
                "REVIEWED_AT",
                "REVIEW_NOTES",
            ]:
                value = previous[
                    source_id
                ].get(
                    field,
                    "",
                )

                if value:
                    record[
                        field
                    ] = value

        records.append(
            record
        )

    frame = pd.DataFrame(
        records,
        columns=VERIFICATION_COLUMNS,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
    )

    return frame


def read_source_verifications(
    manual_dir: Path,
) -> pd.DataFrame:
    path = (
        manual_dir
        / "expert_source_verifications.csv"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=VERIFICATION_COLUMNS
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )


def validate_source_verifications(
    sources: pd.DataFrame,
    verifications: pd.DataFrame,
) -> None:
    missing_columns = set(
        VERIFICATION_COLUMNS
    ).difference(
        verifications.columns
    )

    if missing_columns:
        raise ValueError(
            "expert_source_verifications "
            "is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if verifications.empty:
        return

    identifiers = (
        verifications[
            "SOURCE_ID"
        ]
        .astype(str)
        .str.strip()
    )

    if identifiers.eq("").any():
        raise ValueError(
            "Source verification contains "
            "blank SOURCE_ID values."
        )

    duplicates = identifiers[
        identifiers.duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:
        raise ValueError(
            "Source verification contains "
            "duplicate SOURCE_ID values: "
            f"{sorted(duplicates.unique())}"
        )

    registered_ids = set(
        sources[
            "SOURCE_ID"
        ]
        .astype(str)
        .str.strip()
    )

    unknown = (
        set(
            identifiers
        )
        - registered_ids
    )

    if unknown:
        raise ValueError(
            "Source verification references "
            "unknown sources: "
            f"{sorted(unknown)}"
        )


def build_source_verification_blockers(
    sources: pd.DataFrame,
    verifications: pd.DataFrame,
) -> pd.DataFrame:
    validate_source_verifications(
        sources,
        verifications,
    )

    verification_by_id = {
        str(row["SOURCE_ID"]): row
        for row in verifications.to_dict(
            orient="records"
        )
    }

    blockers: list[
        dict[str, str]
    ] = []

    for source_id in (
        sources[
            "SOURCE_ID"
        ]
        .astype(str)
        .sort_values()
    ):
        if source_id not in verification_by_id:
            blockers.append(
                {
                    "BLOCKER_TYPE": (
                        "missing_source_verification"
                    ),
                    "SIDE": "",
                    "DIMENSION": "",
                    "PLAYER_NAME": "",
                    "DETAIL": source_id,
                }
            )

            continue

        row = verification_by_id[
            source_id
        ]

        automated_status = str(
            row.get(
                "AUTOMATED_STATUS",
                "",
            )
        ).strip()

        review_status = str(
            row.get(
                "REVIEW_STATUS",
                "",
            )
        ).strip()

        reviewed_by = str(
            row.get(
                "REVIEWED_BY",
                "",
            )
        ).strip()

        reviewed_at = str(
            row.get(
                "REVIEWED_AT",
                "",
            )
        ).strip()

        review_notes = str(
            row.get(
                "REVIEW_NOTES",
                "",
            )
        ).strip()

        review_audit_complete = all(
            [
                reviewed_by,
                reviewed_at,
                review_notes,
            ]
        )

        automated_passed = (
            automated_status
            in ACCEPTED_AUTOMATED_STATUSES
        )

        qualified_override = (
            review_status
            == "verified_with_qualification"
            and review_audit_complete
        )

        if (
            not automated_passed
            and not qualified_override
        ):
            blockers.append(
                {
                    "BLOCKER_TYPE": (
                        "automated_source_"
                        "verification_failed"
                    ),
                    "SIDE": "",
                    "DIMENSION": "",
                    "PLAYER_NAME": "",
                    "DETAIL": (
                        f"{source_id}: "
                        f"{automated_status}"
                    ),
                }
            )

        if (
            review_status
            not in ACCEPTED_REVIEW_STATUSES
        ):
            blockers.append(
                {
                    "BLOCKER_TYPE": (
                        "pending_source_review"
                    ),
                    "SIDE": "",
                    "DIMENSION": "",
                    "PLAYER_NAME": "",
                    "DETAIL": (
                        f"{source_id}: "
                        f"{review_status or 'blank'}"
                    ),
                }
            )

        elif (
            review_status
            == "verified_with_qualification"
            and not review_audit_complete
        ):
            blockers.append(
                {
                    "BLOCKER_TYPE": (
                        "incomplete_qualified_"
                        "source_review"
                    ),
                    "SIDE": "",
                    "DIMENSION": "",
                    "PLAYER_NAME": "",
                    "DETAIL": (
                        f"{source_id}: qualified "
                        "review requires reviewer, "
                        "timestamp, and notes"
                    ),
                }
            )

        elif (
            review_status == "verified"
            and not automated_passed
        ):
            blockers.append(
                {
                    "BLOCKER_TYPE": (
                        "invalid_unqualified_"
                        "source_override"
                    ),
                    "SIDE": "",
                    "DIMENSION": "",
                    "PLAYER_NAME": "",
                    "DETAIL": (
                        f"{source_id}: automated "
                        f"status is {automated_status}"
                    ),
                }
            )

    return pd.DataFrame(
        blockers,
        columns=[
            "BLOCKER_TYPE",
            "SIDE",
            "DIMENSION",
            "PLAYER_NAME",
            "DETAIL",
        ],
    )


def validate_verified_claim_sources(
    claims: pd.DataFrame,
    verifications: pd.DataFrame,
) -> None:
    if claims.empty:
        return

    accepted_claims = claims[
        claims[
            "REVIEW_STATUS"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(
            {
                "verified",
                (
                    "verified_with_"
                    "qualification"
                ),
            }
        )
    ]

    if accepted_claims.empty:
        return

    verified_source_ids = set(
        verifications.loc[
            verifications[
                "REVIEW_STATUS"
            ]
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin(
                ACCEPTED_REVIEW_STATUSES
            ),
            "SOURCE_ID",
        ].astype(str)
    )

    claim_source_ids = set(
        accepted_claims[
            "SOURCE_ID"
        ].astype(str)
    )

    unverified = (
        claim_source_ids
        - verified_source_ids
    )

    if unverified:
        raise ValueError(
            "Verified expert claims reference "
            "sources that have not passed "
            "source review: "
            f"{sorted(unverified)}"
        )
