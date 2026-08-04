from __future__ import annotations

from pathlib import Path

import pandas as pd

from goatlab.data.expert_source_verification import (
    read_source_verifications,
    validate_verified_claim_sources,
)
from goatlab.models.expert_evidence import (
    read_expert_evidence,
    validate_expert_evidence,
)
from goatlab.settings import settings


OUTPUT_PATH = Path(
    "docs/EXPERT_CLAIM_REGISTER.md"
)


def escape(
    value: object,
) -> str:
    return (
        str(value)
        .replace(
            "|",
            "\\|",
        )
        .replace(
            "\n",
            " ",
        )
        .strip()
    )


def main() -> None:
    (
        sources,
        claims,
        dimensions,
    ) = read_expert_evidence(
        settings.manual_dir
    )

    verifications = (
        read_source_verifications(
            settings.manual_dir
        )
    )

    validate_expert_evidence(
        sources,
        claims,
        dimensions,
    )

    validate_verified_claim_sources(
        claims,
        verifications,
    )

    accepted = claims[
        claims[
            "REVIEW_STATUS"
        ]
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
    ].copy()

    lines = [
        "# Expert Claim Register",
        "",
        "This register contains claim-level summaries "
        "of published expert analysis.",
        "",
        "It stores paraphrased analytical findings, "
        "supporting locations, confidence, and limitations. "
        "It does not reproduce source articles or video.",
        "",
        "## Summary",
        "",
        f"- Registered claims: {len(claims)}",
        f"- Accepted claims: {len(accepted)}",
        (
            "- Source families represented: "
            f"{accepted.merge(sources[['SOURCE_ID', 'SOURCE_FAMILY']], on='SOURCE_ID')['SOURCE_FAMILY'].nunique()}"
            if not accepted.empty
            else "- Source families represented: 0"
        ),
        "",
        "The Thinking Basketball anchor profiles are "
        "partial-career sources. Their claims use "
        "`career_through_1998` and `career_through_2018`, "
        "not `career`. They are therefore display and "
        "diagnostic evidence only until broader independent "
        "coverage is added.",
        "",
        "## Claims by player and side",
        "",
        "| Player | Side | Claims |",
        "|---|---|---:|",
    ]

    summary = (
        accepted.groupby(
            [
                "PLAYER_NAME",
                "SIDE",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "CLAIMS",
            }
        )
    )

    for row in summary.itertuples(
        index=False
    ):
        lines.append(
            "| "
            f"{escape(row.PLAYER_NAME)} | "
            f"{escape(row.SIDE)} | "
            f"{row.CLAIMS} |"
        )

    lines.extend(
        [
            "",
            "## Accepted claims",
            "",
            "| Claim | Player | Phase | Side | Dimension | Direction | Confidence | Source | Summary | Limitations |",
            "|---|---|---|---|---|---|---:|---|---|---|",
        ]
    )

    for row in accepted.sort_values(
        [
            "PLAYER_NAME",
            "SIDE",
            "DIMENSION",
            "CLAIM_ID",
        ]
    ).itertuples(
        index=False
    ):
        lines.append(
            "| "
            f"{escape(row.CLAIM_ID)} | "
            f"{escape(row.PLAYER_NAME)} | "
            f"{escape(row.CAREER_PHASE)} | "
            f"{escape(row.SIDE)} | "
            f"{escape(row.DIMENSION)} | "
            f"{escape(row.CLAIM_DIRECTION)} | "
            f"{float(row.CONFIDENCE):.2f} | "
            f"{escape(row.SOURCE_ID)} | "
            f"{escape(row.SUMMARY)} | "
            f"{escape(row.LIMITATIONS)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A registered claim is evidence attributed to "
            "one source family. It is not equivalent to "
            "multi-source consensus.",
            "",
            "No partial-career claim is eligible for the "
            "primary model.",
            "",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
