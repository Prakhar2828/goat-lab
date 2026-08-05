from __future__ import annotations

from pathlib import Path

import pandas as pd

from goatlab.models.expert_evidence import (
    read_expert_evidence,
    score_expert_sources,
    validate_expert_evidence,
)
from goatlab.settings import settings

OUTPUT_PATH = Path(
    "docs/EXPERT_SOURCE_REGISTER.md"
)


def _escape(
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


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    lines = [
        "| "
        + " | ".join(
            columns
        )
        + " |",
        "|"
        + "|".join(
            "---"
            for _ in columns
        )
        + "|",
    ]

    for row in frame[
        columns
    ].itertuples(
        index=False,
        name=None,
    ):
        lines.append(
            "| "
            + " | ".join(
                _escape(value)
                for value in row
            )
            + " |"
        )

    return lines


def main() -> None:
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

    scored = score_expert_sources(
        sources
    ).sort_values(
        [
            "SOURCE_FAMILY",
            "PUBLICATION_DATE",
            "SOURCE_ID",
        ]
    )

    family_summary = (
        scored.groupby(
            "SOURCE_FAMILY",
            as_index=False,
        )
        .agg(
            SOURCES=(
                "SOURCE_ID",
                "nunique",
            ),
            ANALYSTS=(
                "ANALYST",
                lambda values: ", ".join(
                    sorted(
                        set(
                            values.astype(str)
                        )
                    )
                ),
            ),
            BEST_TIER=(
                "SOURCE_QUALITY_SCORE",
                lambda values: (
                    "A"
                    if values.max() >= 14
                    else (
                        "B"
                        if values.max() >= 10
                        else (
                            "C"
                            if values.max() >= 6
                            else "excluded"
                        )
                    )
                ),
            ),
            MAX_QUALITY_SCORE=(
                "SOURCE_QUALITY_SCORE",
                "max",
            ),
        )
    )

    lines = [
        "# Expert Source Register",
        "",
        ("This register contains sources reviewed for the "
        "GOAT Lab Expert Film Consensus component."),
        "",
        ("Registration does not mean that every statement "
        "from a source is accepted. Claims are admitted "
        "separately and must identify a supporting location, "
        "confidence, limitation, and review status."),
        "",
        ("Multiple documents from the same analyst or "
        "methodological project share one `SOURCE_FAMILY`. "
        "They are collapsed before consensus so publishing "
        "more articles cannot create artificial independence."),
        "",
        "## Registry summary",
        "",
        f"- Registered source documents: {len(scored)}",
        (
            "- Independent source families: "
            f"{scored['SOURCE_FAMILY'].nunique()}"
        ),
        (
            "- Tier A documents: "
            f"{int(scored['SOURCE_TIER'].eq('A').sum())}"
        ),
        (
            "- Tier B documents: "
            f"{int(scored['SOURCE_TIER'].eq('B').sum())}"
        ),
        (
            "- Tier C documents: "
            f"{int(scored['SOURCE_TIER'].eq('C').sum())}"
        ),
        "",
        "## Source families",
        "",
    ]

    lines.extend(
        _markdown_table(
            family_summary,
            [
                "SOURCE_FAMILY",
                "SOURCES",
                "ANALYSTS",
                "BEST_TIER",
                "MAX_QUALITY_SCORE",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Registered documents",
            "",
        ]
    )

    display = scored[
        [
            "SOURCE_ID",
            "SOURCE_FAMILY",
            "ANALYST",
            "PUBLICATION",
            "TITLE",
            "PUBLICATION_DATE",
            "PLAYER_COVERAGE",
            "PRIMARY_USE",
            "SOURCE_QUALITY_SCORE",
            "SOURCE_TIER",
            "ACCESS_STATUS",
            "PAYWALL_STATUS",
            "URL",
        ]
    ].copy()

    lines.extend(
        _markdown_table(
            display,
            display.columns.tolist(),
        )
    )

    lines.extend(
        [
            "",
            "## Source-quality interpretation",
            "",
            "- Tier A sources may anchor a dimension.",
            "- Tier B sources may corroborate or qualify evidence.",
            "- Tier C sources are contextual and cannot anchor a dimension.",
            "- Excluded sources do not enter consensus.",
            ("- A source tier measures evidence quality, not whether "
            "the source favors either player."),
            "",
            "## Current status",
            "",
            ("The source registry is populated. Player claims and "
            "dimension-level consensus remain separate release gates."),
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
