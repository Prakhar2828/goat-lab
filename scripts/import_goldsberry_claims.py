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

SOURCE_IDS = {
    "GOLD_MJ_SCORING_2020",
    "GOLD_LBJ_ATLAS_2018",
    "GOLD_LBJ_SCORING_RECORD_2023",
}

COLUMN_ORDER = [
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


def make_claim(
    *,
    claim_id: str,
    source_id: str,
    player_name: str,
    career_phase: str,
    season_start: str,
    season_end: str,
    season_type: str,
    dimension: str,
    direction: str,
    strength: int,
    sample_disclosed: bool,
    confidence: float,
    location: str,
    summary: str,
    limitations: str,
) -> dict[str, object]:
    return {
        "CLAIM_ID": claim_id,
        "SOURCE_ID": source_id,
        "PLAYER_NAME": player_name,
        "CAREER_PHASE": career_phase,
        "SEASON_START": season_start,
        "SEASON_END": season_end,
        "SEASON_TYPE": season_type,
        "SIDE": "offense",
        "DIMENSION": dimension,
        "CLAIM_DIRECTION": direction,
        "CLAIM_STRENGTH": strength,
        "EVIDENCE_TYPE": (
            "spatial_scoring_analysis"
        ),
        "FILM_EXAMPLES_PRESENT": False,
        "SAMPLE_SIZE_DISCLOSED": (
            sample_disclosed
        ),
        "CONFIDENCE": confidence,
        "SUPPORTING_LOCATION": location,
        "SUMMARY": summary,
        "LIMITATIONS": limitations,
        "REVIEW_STATUS": (
            "verified_with_qualification"
        ),
    }


def build_claims() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(**kwargs: object) -> None:
        rows.append(
            make_claim(**kwargs)
        )

    # Michael Jordan: 2020 spatial scoring review.
    add(
        claim_id="GOLD_MJ_2020_O_MIDRANGE",
        source_id="GOLD_MJ_SCORING_2020",
        player_name="Michael Jordan",
        career_phase="late_bulls_1996_1998",
        season_start="1996-97",
        season_end="1997-98",
        season_type="all",
        dimension="midrange_creation",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.98,
        location=(
            "Studying Jordan's old shooting data — "
            "1996-97 midrange volume and efficiency"
        ),
        summary=(
            "The spatial data identifies Jordan as an extreme "
            "midrange-volume scorer who also remained highly "
            "efficient despite carrying the central creation burden."
        ),
        limitations=(
            "NBA shot-location coverage begins in 1996-97, so the "
            "quantified sample represents only his final two Bulls "
            "seasons rather than his complete career."
        ),
    )

    add(
        claim_id="GOLD_MJ_2020_O_HALF_COURT",
        source_id="GOLD_MJ_SCORING_2020",
        player_name="Michael Jordan",
        career_phase="late_bulls_1996_1998",
        season_start="1996-97",
        season_end="1997-98",
        season_type="all",
        dimension="half_court_creation",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.93,
        location=(
            "1997 Bullets playoff example and "
            "one-on-one scoring discussion"
        ),
        summary=(
            "Jordan is shown generating efficient half-court offense "
            "against set defenses through pull-ups, drives, post-ups, "
            "fadeaways, and an unpredictable one-on-one move set."
        ),
        limitations=(
            "The article emphasizes scoring possessions and selected "
            "late-career examples; it is not a possession-complete "
            "classification of every half-court action."
        ),
    )

    add(
        claim_id="GOLD_MJ_2020_O_POST",
        source_id="GOLD_MJ_SCORING_2020",
        player_name="Michael Jordan",
        career_phase="late_bulls_1996_1998",
        season_start="1996-97",
        season_end="1997-98",
        season_type="all",
        dimension="post_scoring",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.95,
        location=(
            "Second three-peat post-up, block-location, "
            "and fadeaway discussion"
        ),
        summary=(
            "The late-Bulls shot locations and tactical description "
            "support Jordan's post game as a primary scoring engine, "
            "especially through block catches and fadeaway counters."
        ),
        limitations=(
            "The source does not publish a standardized post-up "
            "possession table, and the quantified location sample "
            "covers only 1996-97 and 1997-98."
        ),
    )

    add(
        claim_id="GOLD_MJ_2020_O_GRAVITY",
        source_id="GOLD_MJ_SCORING_2020",
        player_name="Michael Jordan",
        career_phase="late_bulls_1996_1998",
        season_start="1996-97",
        season_end="1997-98",
        season_type="all",
        dimension="shooting_gravity",
        direction="strength",
        strength=2,
        sample_disclosed=False,
        confidence=0.82,
        location=(
            "Defensive-pressure comparison surrounding "
            "the 1996-97 midrange results"
        ),
        summary=(
            "The analysis describes opponents centering game plans "
            "and elite defenders on Jordan's jumper, indicating "
            "substantial attention created by his shooting threat."
        ),
        limitations=(
            "Defensive attention is described qualitatively; the "
            "article does not provide a direct gravity metric or "
            "off-ball defender-distance sample."
        ),
    )

    add(
        claim_id="GOLD_MJ_2020_O_ROLE_ADAPTATION",
        source_id="GOLD_MJ_SCORING_2020",
        player_name="Michael Jordan",
        career_phase="career_through_1998",
        season_start="1984-85",
        season_end="1997-98",
        season_type="all",
        dimension="role_adaptability",
        direction="major_strength",
        strength=3,
        sample_disclosed=False,
        confidence=0.91,
        location=(
            "Career evolution from rim-attacking leaper "
            "to late-career jump shooter"
        ),
        summary=(
            "The review presents Jordan as changing his scoring "
            "profile as his athleticism declined, replacing more early "
            "rim attacks with refined jump shooting and post creation."
        ),
        limitations=(
            "Early-career shot locations are not available in the "
            "same tracking system, so the cross-career evolution is "
            "partly qualitative rather than fully matched statistically."
        ),
    )

    add(
        claim_id="GOLD_MJ_2020_O_PLAYOFF_RESILIENCE",
        source_id="GOLD_MJ_SCORING_2020",
        player_name="Michael Jordan",
        career_phase="playoffs_1997_1998",
        season_start="1996-97",
        season_end="1997-98",
        season_type="postseason",
        dimension="playoff_resilience",
        direction="strength",
        strength=2,
        sample_disclosed=True,
        confidence=0.80,
        location=(
            "1997 Bullets Game 2 and 1998 postseason "
            "midrange-production sections"
        ),
        summary=(
            "The source documents Jordan sustaining heavy midrange "
            "creation in selected playoff settings, including the "
            "1998 title run when that zone supplied most of his makes."
        ),
        limitations=(
            "This is evidence from selected games and one late-career "
            "postseason, not a complete opponent-adjusted study of "
            "Jordan's playoff resilience across every series."
        ),
    )

    # LeBron James: 2018 offensive atlas.
    add(
        claim_id="GOLD_LBJ_2018_O_RIM",
        source_id="GOLD_LBJ_ATLAS_2018",
        player_name="LeBron James",
        career_phase="career_through_2018",
        season_start="2003-04",
        season_end="2017-18",
        season_type="all",
        dimension="rim_pressure_finishing",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.98,
        location=(
            "How LeBron owned the paint — restricted-area "
            "volume and efficiency progression"
        ),
        summary=(
            "The shot-chart history identifies interior scoring as "
            "LeBron's defining scoring strength, with repeated "
            "league-leading restricted-area production."
        ),
        limitations=(
            "The article ends before his Lakers career and uses "
            "season-level spatial summaries rather than a complete "
            "possession-by-possession finishing taxonomy."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2018_O_ADVANTAGE",
        source_id="GOLD_LBJ_ATLAS_2018",
        player_name="LeBron James",
        career_phase="career_through_2018",
        season_start="2003-04",
        season_end="2017-18",
        season_type="all",
        dimension="advantage_creation",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.96,
        location=(
            "A different kind of 3-point specialist — "
            "help defense and clean-look creation"
        ),
        summary=(
            "LeBron's rim pressure is shown forcing help rotations "
            "that he converts into open perimeter opportunities for "
            "teammates, creating a recurring defensive dilemma."
        ),
        limitations=(
            "The evidence focuses on assisted three-pointers and "
            "spatial pressure, not every downstream advantage or "
            "secondary assist created by the initial rotation."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2018_O_PASSING_EXECUTION",
        source_id="GOLD_LBJ_ATLAS_2018",
        player_name="LeBron James",
        career_phase="career_through_2018",
        season_start="2003-04",
        season_end="2017-18",
        season_type="all",
        dimension="passing_execution",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.94,
        location=(
            "A different kind of 3-point specialist — "
            "three-point assist totals"
        ),
        summary=(
            "The analysis supports elite passing execution through "
            "high-volume, well-timed deliveries to shooters after "
            "LeBron draws interior help."
        ),
        limitations=(
            "Assisted three-pointers capture only one passing outcome "
            "and do not independently grade pass difficulty, accuracy, "
            "or all non-shooting reads."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2018_O_OFF_BALL",
        source_id="GOLD_LBJ_ATLAS_2018",
        player_name="LeBron James",
        career_phase="miami_2010_2014",
        season_start="2010-11",
        season_end="2013-14",
        season_type="all",
        dimension="off_ball_value",
        direction="strength",
        strength=2,
        sample_disclosed=True,
        confidence=0.86,
        location=(
            "Miami efficiency section — lower usage, "
            "assisted threes, and off-ball adjustment"
        ),
        summary=(
            "The Miami phase is presented as evidence that LeBron "
            "could reduce difficult self-created attempts, operate "
            "more off ball, and improve efficiency beside other stars."
        ),
        limitations=(
            "The article does not isolate off-ball possessions or "
            "separate LeBron's movement value from the spacing and "
            "talent supplied by Miami's roster."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2018_O_SCALABILITY",
        source_id="GOLD_LBJ_ATLAS_2018",
        player_name="LeBron James",
        career_phase="miami_2010_2014",
        season_start="2010-11",
        season_end="2013-14",
        season_type="all",
        dimension="scalability",
        direction="strength",
        strength=2,
        sample_disclosed=True,
        confidence=0.88,
        location=(
            "Miami efficiency section — playing with "
            "Wade and Bosh at reduced shot volume"
        ),
        summary=(
            "LeBron retained elite value while lowering shot volume "
            "and sharing creation with Wade and Bosh, supporting his "
            "ability to scale beside other high-level creators."
        ),
        limitations=(
            "The evidence covers one unusually strong roster context "
            "and does not establish identical scalability across every "
            "lineup construction or career phase."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2018_O_ROLE_ADAPTATION",
        source_id="GOLD_LBJ_ATLAS_2018",
        player_name="LeBron James",
        career_phase="career_through_2018",
        season_start="2003-04",
        season_end="2017-18",
        season_type="all",
        dimension="role_adaptability",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.94,
        location=(
            "Rookie, Miami, and second-Cleveland "
            "shot-profile progression"
        ),
        summary=(
            "The article traces LeBron from an inefficient rookie "
            "scorer to an interior force and then a more selective, "
            "versatile creator across changing team environments."
        ),
        limitations=(
            "The review stops in 2018 and therefore cannot evaluate "
            "his later Lakers adaptations or his complete career arc."
        ),
    )

    # LeBron James: 2023 scoring-record shot-chart review.
    add(
        claim_id="GOLD_LBJ_2023_O_RIM",
        source_id="GOLD_LBJ_SCORING_RECORD_2023",
        player_name="LeBron James",
        career_phase="career_through_2023",
        season_start="2003-04",
        season_end="2022-23",
        season_type="all",
        dimension="rim_pressure_finishing",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.98,
        location=(
            "2004-05 through return-to-Cleveland "
            "restricted-area scoring sections"
        ),
        summary=(
            "The twenty-season shot-chart review confirms sustained "
            "interior scoring as the central foundation of LeBron's "
            "career scoring production."
        ),
        limitations=(
            "The article is a scoring-record retrospective through "
            "early 2023 and does not cover the remaining seasons of "
            "his career or classify every finishing attempt."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2023_O_MIDRANGE",
        source_id="GOLD_LBJ_SCORING_RECORD_2023",
        player_name="LeBron James",
        career_phase="career_through_2023",
        season_start="2003-04",
        season_end="2022-23",
        season_type="all",
        dimension="midrange_creation",
        direction="strength",
        strength=2,
        sample_disclosed=True,
        confidence=0.82,
        location=(
            "Miami peak and scoring-record fadeaway "
            "shot-chart discussion"
        ),
        summary=(
            "The review shows meaningful midrange development from "
            "poor rookie efficiency to productive high-leverage use, "
            "including during his Miami peak."
        ),
        limitations=(
            "Midrange shooting is not presented as LeBron's dominant "
            "career weapon, and the article does not provide a single "
            "era-adjusted career midrange grade."
        ),
    )

    add(
        claim_id="GOLD_LBJ_2023_O_ROLE_ADAPTATION",
        source_id="GOLD_LBJ_SCORING_RECORD_2023",
        player_name="LeBron James",
        career_phase="career_through_2023",
        season_start="2003-04",
        season_end="2022-23",
        season_type="all",
        dimension="role_adaptability",
        direction="major_strength",
        strength=3,
        sample_disclosed=True,
        confidence=0.96,
        location=(
            "Rookie-to-Laker shot-chart evolution "
            "and changing jumper-distance sections"
        ),
        summary=(
            "The longitudinal shot charts show LeBron changing shot "
            "selection and scoring method across eras, teams, age, "
            "and the league-wide shift toward three-point shooting."
        ),
        limitations=(
            "The source evaluates scoring adaptation rather than every "
            "offensive responsibility, and its coverage ends during "
            "the 2022-23 season."
        ),
    )

    frame = pd.DataFrame(
        rows,
        columns=COLUMN_ORDER,
    )

    if frame[
        "CLAIM_ID"
    ].duplicated().any():
        duplicates = (
            frame.loc[
                frame[
                    "CLAIM_ID"
                ].duplicated(
                    keep=False
                ),
                "CLAIM_ID",
            ]
            .sort_values()
            .unique()
            .tolist()
        )
        raise ValueError(
            "Duplicate Goldsberry claim IDs: "
            f"{duplicates}"
        )

    return frame


def main() -> None:
    sources = pd.read_csv(
        settings.manual_dir
        / "expert_sources.csv"
    )
    dimensions = pd.read_csv(
        settings.manual_dir
        / "expert_analysis_dimensions.csv"
    )
    existing = pd.read_csv(
        OUTPUT_PATH
    )
    new_claims = build_claims()

    missing_sources = (
        SOURCE_IDS
        - set(
            sources[
                "SOURCE_ID"
            ].astype(str)
        )
    )

    if missing_sources:
        raise ValueError(
            "Missing Goldsberry source rows: "
            f"{sorted(missing_sources)}"
        )

    retained = existing.loc[
        ~existing[
            "SOURCE_ID"
        ].isin(
            SOURCE_IDS
        )
    ].copy()

    combined = pd.concat(
        [
            retained,
            new_claims,
        ],
        ignore_index=True,
    )

    combined = (
        combined[
            COLUMN_ORDER
        ]
        .sort_values(
            [
                "PLAYER_NAME",
                "SIDE",
                "CAREER_PHASE",
                "SOURCE_ID",
                "DIMENSION",
                "CLAIM_ID",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

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
        f"Wrote {len(new_claims)} "
        "Goldsberry claims."
    )
    print(
        f"Total expert claims: "
        f"{len(combined)}"
    )
    print()
    print(
        new_claims.groupby(
            [
                "SOURCE_ID",
                "PLAYER_NAME",
            ]
        )
        .size()
        .rename(
            "CLAIMS"
        )
    )


if __name__ == "__main__":
    main()
