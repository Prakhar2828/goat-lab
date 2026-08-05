from __future__ import annotations

import pandas as pd

from goatlab.models.expert_evidence import (
    validate_expert_evidence,
)
from goatlab.settings import settings

OUTPUT_PATH = (
    settings.manual_dir
    / "expert_claims.csv"
)

SOURCES_PATH = (
    settings.manual_dir
    / "expert_sources.csv"
)

DIMENSIONS_PATH = (
    settings.manual_dir
    / "expert_analysis_dimensions.csv"
)

SOURCE_IDS = {
    "LOWE_GOAT_DEBATE_2020",
    "LOWE_LAKERS_DEFENSE_2020",
}

CLAIM_COLUMNS = [
    "CLAIM_ID",
    "SOURCE_ID",
    "PLAYER_NAME",
    "CAREER_PHASE",
    "SEASON_START",
    "SEASON_END",
    "SEASON_TYPE",
    "SIDE",
    "DIMENSION",
    "CLAIM_DIRECTION",
    "CLAIM_STRENGTH",
    "EVIDENCE_TYPE",
    "FILM_EXAMPLES_PRESENT",
    "SAMPLE_SIZE_DISCLOSED",
    "CONFIDENCE",
    "SUPPORTING_LOCATION",
    "SUMMARY",
    "LIMITATIONS",
    "REVIEW_STATUS",
]


def claim(
    claim_id: str,
    source_id: str,
    player_name: str,
    career_phase: str,
    season_start: str,
    season_end: str,
    season_type: str,
    side: str,
    dimension: str,
    direction: str,
    strength: int,
    confidence: float,
    location: str,
    summary: str,
    limitations: str,
    *,
    evidence_type: str,
    film_examples: bool,
    sample_disclosed: bool,
) -> dict[str, object]:
    return {
        "CLAIM_ID": claim_id,
        "SOURCE_ID": source_id,
        "PLAYER_NAME": player_name,
        "CAREER_PHASE": career_phase,
        "SEASON_START": season_start,
        "SEASON_END": season_end,
        "SEASON_TYPE": season_type,
        "SIDE": side,
        "DIMENSION": dimension,
        "CLAIM_DIRECTION": direction,
        "CLAIM_STRENGTH": strength,
        "EVIDENCE_TYPE": evidence_type,
        "FILM_EXAMPLES_PRESENT": film_examples,
        "SAMPLE_SIZE_DISCLOSED": sample_disclosed,
        "CONFIDENCE": confidence,
        "SUPPORTING_LOCATION": location,
        "SUMMARY": summary,
        "LIMITATIONS": limitations,
        "REVIEW_STATUS": (
            "verified_with_qualification"
        ),
    }


def build_claims() -> list[dict[str, object]]:
    comparative = (
        "comparative_tactical_and_statistical_analysis"
    )
    tactical = (
        "film_based_tactical_analysis"
    )

    claims = [
        claim(
            "LOWE_GOAT_MJ_LATE_CLOCK",
            "LOWE_GOAT_DEBATE_2020",
            "Michael Jordan",
            "career_through_1998",
            "1984-85",
            "1997-98",
            "playoffs",
            "offense",
            "late_clock_creation",
            "major_strength",
            3,
            0.96,
            (
                "Late-game comparison — walk-off shots, "
                "Finals closers, and clutch dossier"
            ),
            (
                "Lowe evaluates Jordan as an unmatched late-game "
                "volume scorer whose efficiency and shot ownership "
                "held up in championship-level possessions."
            ),
            (
                "The article is a retrospective comparison and does "
                "not disclose a possession-complete late-clock sample."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_GOAT_MJ_HALF_COURT",
            "LOWE_GOAT_DEBATE_2020",
            "Michael Jordan",
            "career_through_1998",
            "1984-85",
            "1997-98",
            "all",
            "offense",
            "half_court_creation",
            "major_strength",
            3,
            0.92,
            (
                "Era and advanced-stat comparison — isolation "
                "scoring, usage, and efficiency discussion"
            ),
            (
                "Jordan is presented as the superior isolation scorer "
                "and as a creator who maintained extraordinary "
                "efficiency at the highest scoring burdens."
            ),
            (
                "The article does not classify every half-court "
                "possession and explicitly notes major era differences."
            ),
            evidence_type=comparative,
            film_examples=False,
            sample_disclosed=True,
        ),
        claim(
            "LOWE_GOAT_MJ_TURNOVERS",
            "LOWE_GOAT_DEBATE_2020",
            "Michael Jordan",
            "career_through_1998",
            "1984-85",
            "1997-98",
            "all",
            "offense",
            "turnover_management",
            "major_strength",
            3,
            0.86,
            (
                "Era comparison — usage, passing environment, "
                "and turnover-rate discussion"
            ),
            (
                "Lowe highlights Jordan's unusually low turnover rate "
                "relative to his scoring burden while questioning how "
                "modern help schemes might have changed that result."
            ),
            (
                "The comparison is era-sensitive and does not provide "
                "a role-adjusted possession model for turnovers."
            ),
            evidence_type=comparative,
            film_examples=False,
            sample_disclosed=True,
        ),
        claim(
            "LOWE_GOAT_MJ_PLAYOFF_RESILIENCE",
            "LOWE_GOAT_DEBATE_2020",
            "Michael Jordan",
            "career_through_1998",
            "1984-85",
            "1997-98",
            "playoffs",
            "offense",
            "playoff_resilience",
            "major_strength",
            3,
            0.95,
            (
                "Jordan postseason review — contested series, "
                "late-game scoring, and six title runs"
            ),
            (
                "The article presents Jordan's scoring and decision "
                "making as exceptionally resilient across high-leverage "
                "series and repeated championship runs."
            ),
            (
                "Team outcomes and narrative memory are intertwined "
                "with the individual evaluation in this comparison."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_GOAT_LBJ_LATE_CLOCK",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "playoffs",
            "offense",
            "late_clock_creation",
            "major_strength",
            3,
            0.94,
            (
                "LeBron high-stakes review — 2013, 2016, "
                "2018, and 2020 closing possessions"
            ),
            (
                "Lowe finds a large body of late-game scoring and "
                "playmaking that approaches Jordan's record even when "
                "LeBron chooses the pass dictated by the defense."
            ),
            (
                "The article also documents misses and turnovers and "
                "does not provide a complete late-clock possession file."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_GOAT_LBJ_PASSING_EXECUTION",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "all",
            "offense",
            "passing_execution",
            "major_strength",
            3,
            0.93,
            (
                "Crunch-time decision discussion — right-play "
                "framing and 2018 Finals pass to George Hill"
            ),
            (
                "LeBron is evaluated as an elite executor who can make "
                "the correct high-difficulty pass when layered help "
                "removes a direct scoring lane."
            ),
            (
                "The source uses selected high-leverage examples rather "
                "than a complete pass-location and accuracy sample."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_GOAT_LBJ_PLAYOFF_RESILIENCE",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "playoffs",
            "offense",
            "playoff_resilience",
            "strength",
            2,
            0.92,
            (
                "LeBron postseason review — Finals losses, "
                "2011 failure, and later elimination-game peaks"
            ),
            (
                "The source supports exceptional playoff resilience "
                "across many series while preserving the 2011 Finals "
                "as material counterevidence."
            ),
            (
                "This is deliberately coded as a qualified strength "
                "because the article emphasizes both dominance and "
                "major postseason failures."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=True,
        ),
        claim(
            "LOWE_GOAT_LBJ_RIM_PRESSURE",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "playoffs",
            "offense",
            "rim_pressure_finishing",
            "major_strength",
            3,
            0.86,
            (
                "High-stakes scoring examples — bulldozing "
                "layups and layered-help discussion"
            ),
            (
                "LeBron's power driving and finishing are shown as "
                "reliable sources of late-game pressure even against "
                "multiple defenders occupying his driving corridors."
            ),
            (
                "The article is not a career shot-chart study and "
                "focuses primarily on memorable postseason possessions."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_GOAT_LBJ_ADVANTAGE_CREATION",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "all",
            "offense",
            "advantage_creation",
            "major_strength",
            3,
            0.91,
            (
                "Era comparison — one-on-team help, "
                "drive-and-kick play, and space occupation"
            ),
            (
                "The article describes modern defenses loading several "
                "helpers toward LeBron, creating passing advantages "
                "when he reads the occupied space correctly."
            ),
            (
                "Era rules and roster spacing complicate direct "
                "comparison with Jordan's isolation environment."
            ),
            evidence_type=comparative,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_GOAT_LBJ_RIM_DETERRENCE",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "all",
            "defense",
            "rim_protection",
            "strength",
            2,
            0.81,
            (
                "Advanced comparison — blocks, rim deterrence, "
                "and defensive weapon discussion"
            ),
            (
                "Lowe characterizes LeBron as a meaningful rim "
                "deterrent whose defensive value extends beyond "
                "traditional wing assignments."
            ),
            (
                "The article provides no possession-level rim contest "
                "sample and does not isolate LeBron from team context."
            ),
            evidence_type=comparative,
            film_examples=False,
            sample_disclosed=True,
        ),
        claim(
            "LOWE_GOAT_LBJ_POSITIONAL_VERSATILITY",
            "LOWE_GOAT_DEBATE_2020",
            "LeBron James",
            "career_through_2020",
            "2003-04",
            "2019-20",
            "all",
            "defense",
            "positional_versatility",
            "major_strength",
            3,
            0.88,
            (
                "Advanced comparison — positional flexibility "
                "and defensive-role contrast"
            ),
            (
                "LeBron is evaluated as a distinct defensive weapon "
                "whose size and mobility provide greater positional "
                "flexibility across assignments."
            ),
            (
                "The comparison is concise and does not catalogue "
                "matchup frequency across every team and career phase."
            ),
            evidence_type=comparative,
            film_examples=False,
            sample_disclosed=True,
        ),
        claim(
            "LOWE_LAL_LBJ_HELP_POSITIONING",
            "LOWE_LAKERS_DEFENSE_2020",
            "LeBron James",
            "lakers_2019_20",
            "2019-20",
            "2019-20",
            "all",
            "defense",
            "help_positioning",
            "major_strength",
            3,
            0.94,
            (
                "Lakers film review — strong-side help, "
                "paint deterrence, and corner-priority examples"
            ),
            (
                "LeBron repeatedly reads which threat requires help, "
                "occupies the correct space, and deters drives without "
                "automatically abandoning more dangerous shooters."
            ),
            (
                "The evidence covers one title season within an elite "
                "team defense anchored by Anthony Davis."
            ),
            evidence_type=tactical,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_LAL_LBJ_ROTATION_TIMING",
            "LOWE_LAKERS_DEFENSE_2020",
            "LeBron James",
            "lakers_2019_20",
            "2019-20",
            "2019-20",
            "all",
            "defense",
            "rotation_timing",
            "major_strength",
            3,
            0.94,
            (
                "Lakers film review — Harris and Grant "
                "rotation chain against Denver"
            ),
            (
                "The source shows LeBron anticipating the next pass, "
                "completing sequential rotations, and preserving the "
                "team's coverage after an initial defensive shift."
            ),
            (
                "The examples are selected from a successful postseason "
                "run rather than a randomized season-long film sample."
            ),
            evidence_type=tactical,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_LAL_LBJ_PASSING_LANES",
            "LOWE_LAKERS_DEFENSE_2020",
            "LeBron James",
            "lakers_2019_20",
            "2019-20",
            "2019-20",
            "all",
            "defense",
            "passing_lane_disruption",
            "strength",
            2,
            0.87,
            (
                "Regular-season defensive review — ferocious "
                "rotations and passing-lane activity"
            ),
            (
                "LeBron's renewed activity is described as disrupting "
                "passing lanes and shrinking driving windows within "
                "the Lakers' size-based scheme."
            ),
            (
                "The article does not disclose a complete deflection "
                "sample or separate gambling costs from successful plays."
            ),
            evidence_type=tactical,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_LAL_LBJ_COMMUNICATION",
            "LOWE_LAKERS_DEFENSE_2020",
            "LeBron James",
            "lakers_2019_20",
            "2019-20",
            "2019-20",
            "all",
            "defense",
            "communication_recognition",
            "major_strength",
            3,
            0.96,
            (
                "Lakers scheme development — film sessions, "
                "coverage proposals, and teammate corrections"
            ),
            (
                "LeBron is shown recognizing mistakes, accepting "
                "corrections, directing teammates, and proposing "
                "opponent-specific coverage changes."
            ),
            (
                "Leadership testimony comes primarily from coaches, "
                "teammates, and people around the 2019-20 Lakers."
            ),
            evidence_type=tactical,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_LAL_LBJ_EFFORT",
            "LOWE_LAKERS_DEFENSE_2020",
            "LeBron James",
            "lakers_2019_20",
            "2019-20",
            "2019-20",
            "all",
            "defense",
            "effort_consistency",
            "major_strength",
            3,
            0.91,
            (
                "Season narrative — training-camp buy-in, "
                "regular-season engagement, and bubble defense"
            ),
            (
                "The source documents a sustained recalibration in "
                "LeBron's defensive effort from training camp through "
                "the championship run."
            ),
            (
                "This supports consistency within one season, not "
                "consistency across LeBron's entire career."
            ),
            evidence_type=tactical,
            film_examples=True,
            sample_disclosed=False,
        ),
        claim(
            "LOWE_LAL_LBJ_PLAYOFF_ADAPTABILITY",
            "LOWE_LAKERS_DEFENSE_2020",
            "LeBron James",
            "lakers_2019_20",
            "2019-20",
            "2019-20",
            "playoffs",
            "defense",
            "playoff_adaptability",
            "major_strength",
            3,
            0.95,
            (
                "Postseason scheme review — Portland, Houston, "
                "Denver, and Miami coverage changes"
            ),
            (
                "LeBron executes changing help, switching, trapping, "
                "and matchup rules as the Lakers alter their defense "
                "for four different playoff opponents."
            ),
            (
                "The article evaluates LeBron inside a highly connected "
                "team scheme and does not isolate every adjustment's "
                "individual effect."
            ),
            evidence_type=tactical,
            film_examples=True,
            sample_disclosed=True,
        ),
    ]

    return claims


def main() -> None:
    sources = pd.read_csv(
        SOURCES_PATH
    )
    existing = pd.read_csv(
        OUTPUT_PATH
    )
    dimensions = pd.read_csv(
        DIMENSIONS_PATH
    )

    registered = set(
        sources[
            "SOURCE_ID"
        ].astype(str)
    )

    missing_sources = (
        SOURCE_IDS
        - registered
    )

    if missing_sources:
        raise ValueError(
            "Missing registered Lowe sources: "
            f"{sorted(missing_sources)}"
        )

    additions = pd.DataFrame(
        build_claims(),
        columns=CLAIM_COLUMNS,
    )

    retained = existing[
        ~existing[
            "SOURCE_ID"
        ].astype(str).isin(
            SOURCE_IDS
        )
    ].copy()

    combined = pd.concat(
        [
            retained,
            additions,
        ],
        ignore_index=True,
    )

    combined = combined[
        CLAIM_COLUMNS
    ]

    validate_expert_evidence(
        sources,
        combined,
        dimensions,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        lineterminator="\n",
    )

    print(
        f"Wrote {len(additions)} Lowe claims."
    )
    print(
        f"Total expert claims: {len(combined)}"
    )
    print()
    print(
        additions.groupby(
            [
                "SOURCE_ID",
                "PLAYER_NAME",
                "SIDE",
            ]
        )
        .size()
        .rename("CLAIMS")
    )


if __name__ == "__main__":
    main()
