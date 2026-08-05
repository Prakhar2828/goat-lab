from __future__ import annotations

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

OUTPUT_PATH = (
    settings.manual_dir
    / "expert_claims.csv"
)

SOURCE_IDS = {
    "TB_MJ_BACKPICKS_2018",
    "TB_LBJ_BACKPICKS_2018",
}


def make_claim(
    *,
    claim_id: str,
    source_id: str,
    player_name: str,
    career_phase: str,
    season_start: str,
    season_end: str,
    side: str,
    dimension: str,
    direction: str,
    strength: int,
    evidence_type: str,
    film_examples: bool,
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
        "SEASON_TYPE": "all",
        "SIDE": side,
        "DIMENSION": dimension,
        "CLAIM_DIRECTION": direction,
        "CLAIM_STRENGTH": strength,
        "EVIDENCE_TYPE": evidence_type,
        "FILM_EXAMPLES_PRESENT": (
            film_examples
        ),
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


claims: list[dict[str, object]] = []


def add_jordan(
    suffix: str,
    side: str,
    dimension: str,
    direction: str,
    strength: int,
    confidence: float,
    location: str,
    summary: str,
    limitations: str,
    *,
    evidence_type: str = (
        "film_and_statistical_profile"
    ),
    film_examples: bool = True,
    sample_disclosed: bool = False,
) -> None:
    claims.append(
        make_claim(
            claim_id=f"TB_MJ_2018_{suffix}",
            source_id=(
                "TB_MJ_BACKPICKS_2018"
            ),
            player_name="Michael Jordan",
            career_phase=(
                "career_through_1998"
            ),
            season_start="1984-85",
            season_end="1997-98",
            side=side,
            dimension=dimension,
            direction=direction,
            strength=strength,
            evidence_type=evidence_type,
            film_examples=film_examples,
            sample_disclosed=(
                sample_disclosed
            ),
            confidence=confidence,
            location=location,
            summary=summary,
            limitations=limitations,
        )
    )


def add_lebron(
    suffix: str,
    side: str,
    dimension: str,
    direction: str,
    strength: int,
    confidence: float,
    location: str,
    summary: str,
    limitations: str,
    *,
    evidence_type: str = (
        "film_and_statistical_profile"
    ),
    film_examples: bool = True,
    sample_disclosed: bool = False,
) -> None:
    claims.append(
        make_claim(
            claim_id=f"TB_LBJ_2018_{suffix}",
            source_id=(
                "TB_LBJ_BACKPICKS_2018"
            ),
            player_name="LeBron James",
            career_phase=(
                "career_through_2018"
            ),
            season_start="2003-04",
            season_end="2017-18",
            side=side,
            dimension=dimension,
            direction=direction,
            strength=strength,
            evidence_type=evidence_type,
            film_examples=film_examples,
            sample_disclosed=(
                sample_disclosed
            ),
            confidence=confidence,
            location=location,
            summary=summary,
            limitations=limitations,
        )
    )


# -------------------------------------------------------------------
# Michael Jordan: offense
# -------------------------------------------------------------------

add_jordan(
    "O_RIM_PRESSURE",
    "offense",
    "rim_pressure_finishing",
    "major_strength",
    3,
    0.96,
    (
        "Scouting Report — first-step, "
        "open-space attack, and rim-finishing discussion"
    ),
    (
        "Jordan is presented as an exceptional space attacker "
        "whose first step, elevation, body control, and finishing "
        "created elite pressure around the basket."
    ),
    (
        "The profile provides illustrative possessions but does "
        "not disclose a complete career sample of rim attempts."
    ),
)

add_jordan(
    "O_HALF_COURT",
    "offense",
    "half_court_creation",
    "major_strength",
    3,
    0.97,
    (
        "Scouting Report — apex scoring weapon "
        "and multi-context creation discussion"
    ),
    (
        "The source describes Jordan as an exceptionally complete "
        "half-court scorer through isolation, post play, pick-and-roll, "
        "off-ball actions, footwork, and pull-up creation."
    ),
    (
        "The source is a retrospective expert profile rather than "
        "a possession-complete half-court classification study."
    ),
)

add_jordan(
    "O_MIDRANGE",
    "offense",
    "midrange_creation",
    "major_strength",
    3,
    0.97,
    (
        "Scouting Report — midrange accuracy and "
        "fadeaway development; footnote 1"
    ),
    (
        "Jordan's elevation, footwork, separation moves, and accuracy "
        "are evaluated as an elite and durable midrange creation package."
    ),
    (
        "The disclosed sample covers roughly one hundred longer "
        "jump shots and does not represent every season or shot type."
    ),
    sample_disclosed=True,
)

add_jordan(
    "O_SHOOTING_GRAVITY",
    "offense",
    "shooting_gravity",
    "strength",
    2,
    0.84,
    (
        "Scouting Report — jumper, staggered-screen, "
        "catch-and-shoot, and penetration discussion"
    ),
    (
        "Jordan's jumper and decisive catch attacks created meaningful "
        "off-ball attention and prevented defenders from leaving him unattended."
    ),
    (
        "The source does not publish a direct gravity metric or a "
        "systematic comparison with modern high-volume three-point threats."
    ),
)

add_jordan(
    "O_POST_SCORING",
    "offense",
    "post_scoring",
    "strength",
    3,
    0.90,
    (
        "Scouting Report — complete scoring weapon, "
        "fadeaway, footwork, and late-career physicality"
    ),
    (
        "The profile identifies post scoring, footwork, strength, "
        "and the fadeaway as major parts of Jordan's half-court value."
    ),
    (
        "No separate post-up frequency or efficiency sample is "
        "disclosed in the article."
    ),
)

add_jordan(
    "O_PASSING_VISION",
    "offense",
    "passing_vision",
    "limitation",
    2,
    0.90,
    (
        "Scouting Report — missed creation opportunities, "
        "court-vision limitations, and footnotes 2-4"
    ),
    (
        "Jordan improved substantially as a decision-maker, but the "
        "source identifies persistent limitations in recognizing some "
        "high-value passing opportunities."
    ),
    (
        "The limitation was strongest early in his career and should "
        "not be applied equally to his mature championship seasons."
    ),
    sample_disclosed=True,
)

add_jordan(
    "O_PASSING_EXECUTION",
    "offense",
    "passing_execution",
    "strength",
    2,
    0.82,
    (
        "Scouting Report — above-average passing, "
        "open-court delivery, and footnote 4"
    ),
    (
        "Jordan developed into an above-average passer capable of "
        "accurate high-value completions and advanced open-court deliveries."
    ),
    (
        "The source's tracked passing profile remains below some "
        "other elite perimeter creators and is not full-career play-by-play."
    ),
    sample_disclosed=True,
)

add_jordan(
    "O_ADVANTAGE_CREATION",
    "offense",
    "advantage_creation",
    "major_strength",
    3,
    0.96,
    (
        "Scouting Report — decision-making improvement "
        "and 1989-97 creation-rate discussion"
    ),
    (
        "After improving shot selection and floor reads, Jordan "
        "generated elite historical creation rates while preserving "
        "his extraordinary scoring burden."
    ),
    (
        "The strongest creation evidence applies primarily to "
        "1989-97 rather than his earliest seasons."
    ),
    sample_disclosed=True,
)

add_jordan(
    "O_TURNOVER_CONTROL",
    "offense",
    "turnover_management",
    "major_strength",
    3,
    0.98,
    (
        "Key Stats and Trends and Impact Evaluation — "
        "ball security and team turnover discussion"
    ),
    (
        "Jordan combined extreme scoring and creation volume with "
        "exceptionally low turnover rates and decisive offensive actions."
    ),
    (
        "Team turnover results also reflect coaching, teammates, "
        "scheme, and lineup construction."
    ),
    evidence_type=(
        "film_and_statistical_context"
    ),
)

add_jordan(
    "O_TRANSITION",
    "offense",
    "transition_offense",
    "strength",
    2,
    0.84,
    (
        "Scouting Report — open-court speed, "
        "rebounding, and transition passing examples"
    ),
    (
        "Jordan's acceleration, ball pursuit, finishing, and passing "
        "made him a high-value open-court offensive player."
    ),
    (
        "The article does not disclose a systematic transition "
        "frequency or efficiency sample."
    ),
)

add_jordan(
    "O_OFF_BALL",
    "offense",
    "off_ball_value",
    "major_strength",
    3,
    0.95,
    (
        "Scouting Report — Doug Collins off-ball "
        "actions and Impact Evaluation portability discussion"
    ),
    (
        "Jordan generated substantial value through curls, flares, "
        "catch-and-shoot actions, cuts, quick attacks, and low time of possession."
    ),
    (
        "The source does not quantify off-ball value independently "
        "from his broader scoring impact."
    ),
)

add_jordan(
    "O_SCALABILITY",
    "offense",
    "scalability",
    "mixed",
    2,
    0.80,
    (
        "Impact Evaluation — hybrid on/off-ball value "
        "and triangle-dependence discussion"
    ),
    (
        "Jordan's hybrid on-ball and off-ball game reduced redundancy "
        "beside other creators, but the source retains reservations "
        "about how much the triangle improved his decision-making."
    ),
    (
        "This is a counterfactual portability judgment rather than "
        "a directly observed experiment across many franchises."
    ),
)

add_jordan(
    "O_PLAYOFF_RESILIENCE",
    "offense",
    "playoff_resilience",
    "major_strength",
    3,
    0.96,
    (
        "Impact Evaluation — postseason scoring volume, "
        "creation, efficiency, and team offense"
    ),
    (
        "Jordan increased postseason scoring volume and creation "
        "without a meaningful efficiency collapse while leading "
        "historically strong playoff offenses."
    ),
    (
        "The claim uses aggregate postseason evidence and does not "
        "separately grade every opponent coverage or series."
    ),
    evidence_type=(
        "statistical_and_contextual_profile"
    ),
    film_examples=False,
)

add_jordan(
    "O_ROLE_ADAPTABILITY",
    "offense",
    "role_adaptability",
    "major_strength",
    3,
    0.93,
    (
        "Scouting Report — decision-making evolution "
        "and second-three-peat skill adaptation"
    ),
    (
        "Jordan evolved from an athletic, shoot-first attacker into "
        "a more selective creator who relied increasingly on strength, "
        "footwork, positioning, and midrange craft."
    ),
    (
        "The profile ends with his 1998 retirement and does not "
        "evaluate the Washington phase."
    ),
)

add_jordan(
    "O_PORTABILITY",
    "offense",
    "offensive_portability",
    "strength",
    2,
    0.78,
    (
        "Impact Evaluation — hybrid role, jumper, "
        "penetration, and plug-and-play reservations"
    ),
    (
        "Jordan's shooting, cutting, quick attacks, and self-creation "
        "suggest strong portability across roster types."
    ),
    (
        "The source explicitly raises uncertainty about how much "
        "Chicago's movement system improved his decision quality."
    ),
)


# -------------------------------------------------------------------
# Michael Jordan: defense
# -------------------------------------------------------------------

add_jordan(
    "D_POINT_OF_ATTACK",
    "defense",
    "point_of_attack",
    "strength",
    3,
    0.90,
    (
        "Scouting Report — 1988 improvement, "
        "point-guard containment, and Finals pressure"
    ),
    (
        "At his defensive peak, Jordan used quickness, size, "
        "positioning, and improved footwork to contain elite perimeter creators."
    ),
    (
        "The source distinguishes this peak from weaker early-career "
        "on-ball defense and lower later-career activity."
    ),
)

add_jordan(
    "D_SCREEN_NAVIGATION",
    "defense",
    "screen_navigation",
    "strength",
    3,
    0.90,
    (
        "Scouting Report — hedge, screen navigation, "
        "ball denial, and second-three-peat strength"
    ),
    (
        "Screen navigation is identified as a clear strength, "
        "supported by quickness early and added strength later."
    ),
    (
        "No complete screen-action sample or matchup-adjusted "
        "success rate is disclosed."
    ),
)

add_jordan(
    "D_BALL_DENIAL",
    "defense",
    "ball_denial",
    "major_strength",
    3,
    0.93,
    (
        "Scouting Report — full-throttle ball denial "
        "and passing-lane shutdown discussion"
    ),
    (
        "Jordan's anticipation, positioning, motor, and hands made "
        "him an exceptional denial defender when fully engaged."
    ),
    (
        "The source also reports that his overall defensive "
        "involvement varied with workload and career stage."
    ),
)

add_jordan(
    "D_HELP_POSITIONING",
    "defense",
    "help_positioning",
    "strength",
    2,
    0.82,
    (
        "Scouting Report — improved rotations, "
        "lane help, and late-career awareness"
    ),
    (
        "Jordan's help positioning and awareness improved markedly, "
        "allowing him to disrupt penetration and create positive team-defense value."
    ),
    (
        "His help style remained aggressive and could be punished "
        "by strong passing or incorrect threat prioritization."
    ),
)

add_jordan(
    "D_ROTATION_TIMING",
    "defense",
    "rotation_timing",
    "mixed",
    2,
    0.88,
    (
        "Scouting Report — improved rotations "
        "contrasted with high defensive error rates"
    ),
    (
        "Jordan developed sharper reads and attentive rotations, "
        "but aggressive decisions continued to generate costly errors."
    ),
    (
        "The article's tracked error estimate covers selected film "
        "rather than every defensive possession."
    ),
    sample_disclosed=True,
)

add_jordan(
    "D_PASSING_LANES",
    "defense",
    "passing_lane_disruption",
    "major_strength",
    3,
    0.92,
    (
        "Scouting Report — steal attempts, ambushes, "
        "lane denial, and disruptive hands"
    ),
    (
        "Jordan's anticipation, hand speed, and ability to disguise "
        "steal attempts created unusually high passing-lane disruption."
    ),
    (
        "The same aggressive style also produced failed gambles, "
        "so disruption cannot be evaluated without error control."
    ),
)

add_jordan(
    "D_RIM_PROTECTION",
    "defense",
    "rim_protection",
    "limitation",
    2,
    0.88,
    (
        "Scouting Report — early rim rotations, "
        "vertical-paint limitations, and ambush blocks"
    ),
    (
        "Jordan could create occasional high-value blocks and steals "
        "near the basket but was not a consistent vertical paint deterrent."
    ),
    (
        "This limitation is partly positional; guard rim protection "
        "should not be judged against center expectations without adjustment."
    ),
)

add_jordan(
    "D_SWITCHABILITY",
    "defense",
    "switchability",
    "strength",
    2,
    0.82,
    (
        "Scouting Report footnote 5 — "
        "Jordan-Pippen switching and cross-match coverage"
    ),
    (
        "Jordan's guard and wing flexibility allowed valuable switching "
        "and transition cross-matching alongside Scottie Pippen."
    ),
    (
        "His switchability did not extend reliably to large interior "
        "players and was enhanced by Pippen's complementary versatility."
    ),
)

add_jordan(
    "D_POSITIONAL_VERSATILITY",
    "defense",
    "positional_versatility",
    "limitation",
    2,
    0.93,
    (
        "Scouting Report — size limitations "
        "against big players and comparison with larger defenders"
    ),
    (
        "Jordan was highly effective across guard and wing assignments "
        "but lacked the size to provide true all-position coverage."
    ),
    (
        "The source compares positional range qualitatively rather "
        "than through a complete matchup database."
    ),
)

add_jordan(
    "D_TRANSITION",
    "defense",
    "transition_defense",
    "limitation",
    2,
    0.87,
    (
        "Scouting Report — transition awareness "
        "and threat-recognition errors"
    ),
    (
        "The source identifies transition awareness as a recurring "
        "weakness, including misreading threats ahead of and behind the play."
    ),
    (
        "Illustrative examples do not establish the full frequency "
        "of transition mistakes across his career."
    ),
)

add_jordan(
    "D_REBOUNDING",
    "defense",
    "defensive_rebounding",
    "strength",
    3,
    0.92,
    (
        "Scouting Report — rebounding quickness "
        "and 1988 defensive-rebounding improvement"
    ),
    (
        "Jordan's quickness, elevation, pursuit, and reduced leaking "
        "made him an unusually valuable rebounder for a guard."
    ),
    (
        "The source does not fully separate individual rebounding "
        "from team scheme and lineup responsibilities."
    ),
)

add_jordan(
    "D_GAMBLING_CONTROL",
    "defense",
    "gambling_error_control",
    "limitation",
    3,
    0.96,
    (
        "Key Stats and Trends and Scouting Report — "
        "high-risk style and defensive error-rate discussion"
    ),
    (
        "Jordan's disruptive aggression produced spectacular plays "
        "but also frequent failed gambles and above-average defensive errors."
    ),
    (
        "The error estimate is based on the analyst's tracked sample "
        "and should not be treated as a full official league dataset."
    ),
    sample_disclosed=True,
)

add_jordan(
    "D_EFFORT_CONSISTENCY",
    "defense",
    "effort_consistency",
    "mixed",
    2,
    0.90,
    (
        "Scouting Report — 1988-89 motor peak "
        "and later defensive-activity reduction"
    ),
    (
        "Jordan reached an exceptional activity level at his defensive "
        "peak but reduced involvement as offensive workload and age increased."
    ),
    (
        "Changes in activity may reflect strategic energy conservation "
        "rather than a simple lack of effort."
    ),
)

add_jordan(
    "D_PLAYOFF_ADAPTABILITY",
    "defense",
    "playoff_adaptability",
    "strength",
    2,
    0.84,
    (
        "Scouting Report — 1991 Finals pressure, "
        "post defense, denial, and later-career adjustments"
    ),
    (
        "Jordan demonstrated the ability to alter pressure points, "
        "deny actions, navigate screens, and use strength or anticipation "
        "against playoff matchups."
    ),
    (
        "The source also documents failed steal attempts and does "
        "not provide a series-by-series defensive grading."
    ),
)


# -------------------------------------------------------------------
# LeBron James: offense
# -------------------------------------------------------------------

add_lebron(
    "O_RIM_PRESSURE",
    "offense",
    "rim_pressure_finishing",
    "major_strength",
    3,
    0.97,
    (
        "Scouting Report — speed, power, "
        "contact finishing, angles, and cutting"
    ),
    (
        "LeBron's combination of size, speed, strength, and acceleration "
        "created historically powerful pressure at the basket."
    ),
    (
        "The source profile stops in 2018 and therefore does not "
        "cover his complete late-career decline or adaptation."
    ),
)

add_lebron(
    "O_HALF_COURT",
    "offense",
    "half_court_creation",
    "major_strength",
    3,
    0.95,
    (
        "Scouting Report and Impact Evaluation — "
        "scoring, passing, floor raising, and offensive orchestration"
    ),
    (
        "LeBron is evaluated as an all-time half-court engine who "
        "simultaneously supplies scoring pressure and high-volume creation."
    ),
    (
        "His historically high ball dominance complicates separation "
        "of personal creation from system dependence."
    ),
)

add_lebron(
    "O_SHOOTING_GRAVITY",
    "offense",
    "shooting_gravity",
    "strength",
    2,
    0.86,
    (
        "Scouting Report — improved three-point shot "
        "and Impact Evaluation spot-up reservations"
    ),
    (
        "Improved three-point shooting forced defenses to respect "
        "LeBron outside the paint and increased his value beside other creators."
    ),
    (
        "The source retains reservations about his spot-up shooting "
        "and does not equate his gravity with elite movement shooters."
    ),
)

add_lebron(
    "O_POST_SCORING",
    "offense",
    "post_scoring",
    "strength",
    3,
    0.90,
    (
        "Scouting Report — 2012 interior shift "
        "and Impact Evaluation lineup versatility"
    ),
    (
        "LeBron's move toward interior and post offense increased "
        "shot quality, efficiency, and usefulness beside other ball handlers."
    ),
    (
        "The article does not provide a complete post-up possession "
        "sample across every season."
    ),
)

add_lebron(
    "O_PASSING_VISION",
    "offense",
    "passing_vision",
    "major_strength",
    3,
    0.99,
    (
        "Scouting Report — skip passing, evolved court vision, "
        "creation rates, and tracked high-leverage opportunities"
    ),
    (
        "LeBron's weak-side recognition, skip passing, and ability "
        "to identify high-leverage opportunities are evaluated among "
        "the strongest ever for a scoring centerpiece."
    ),
    (
        "The tracked comparison stops in 2017 and is based on the "
        "analyst's possession samples rather than official league labels."
    ),
    sample_disclosed=True,
)

add_lebron(
    "O_PASSING_EXECUTION",
    "offense",
    "passing_execution",
    "major_strength",
    3,
    0.94,
    (
        "Scouting Report — pass velocity, quality-pass frequency, "
        "completion rate, and inaccurate-pass discussion"
    ),
    (
        "LeBron consistently executed difficult cross-court and "
        "high-leverage passes at elite frequency and accuracy."
    ),
    (
        "The source also identifies an elevated rate of passes that "
        "were too hard, inaccurate, or directed into excessive traffic."
    ),
    sample_disclosed=True,
)

add_lebron(
    "O_ADVANTAGE_CREATION",
    "offense",
    "advantage_creation",
    "major_strength",
    3,
    0.98,
    (
        "Scouting Report — creation-rate increase "
        "and high-leverage passing sample"
    ),
    (
        "LeBron's scoring pressure and evolved passing generated "
        "creation rates near the highest levels estimated by the source."
    ),
    (
        "The exact creation metric is analyst-defined and should "
        "remain separate from official box assists."
    ),
    sample_disclosed=True,
)

add_lebron(
    "O_TRANSITION",
    "offense",
    "transition_offense",
    "major_strength",
    3,
    0.99,
    (
        "Scouting Report — one-man fast break, "
        "transition frequency, and efficiency discussion"
    ),
    (
        "LeBron is evaluated as an extraordinary transition engine "
        "through speed, strength, finishing, and passing."
    ),
    (
        "The profile's transition statistics end at the article's "
        "2018 publication date."
    ),
)

add_lebron(
    "O_OFF_BALL",
    "offense",
    "off_ball_value",
    "strength",
    2,
    0.84,
    (
        "Scouting Report — backcuts and interior play; "
        "Impact Evaluation — cuts, post play, and offensive rebounding"
    ),
    (
        "LeBron added meaningful off-ball value through cutting, "
        "post positioning, finishing, and offensive rebounding."
    ),
    (
        "His offense remained highly ball-dominant, and the source "
        "does not present him as an elite full-time off-ball mover."
    ),
)

add_lebron(
    "O_SCALABILITY",
    "offense",
    "scalability",
    "strength",
    3,
    0.91,
    (
        "Impact Evaluation — performance beside Wade and Irving, "
        "role reduction, and traditional-post-player caveat"
    ),
    (
        "LeBron retained substantial value beside other ball-dominant "
        "stars because of his defense, post game, cutting, rebounding, "
        "and ability to reduce his on-ball role."
    ),
    (
        "The source questions fit beside a traditional low-post scorer "
        "and identifies diminishing returns from extreme ball dominance."
    ),
)

add_lebron(
    "O_PLAYOFF_RESILIENCE",
    "offense",
    "playoff_resilience",
    "major_strength",
    3,
    0.92,
    (
        "Impact Evaluation — postseason statistical profile "
        "and elite offense discussion"
    ),
    (
        "The source evaluates LeBron's postseason scoring and creation "
        "portfolio as historically elite across Cleveland and Miami."
    ),
    (
        "This is an aggregate postseason conclusion rather than a "
        "coverage-specific grade for every playoff series."
    ),
    evidence_type=(
        "statistical_and_contextual_profile"
    ),
    film_examples=False,
)

add_lebron(
    "O_ROLE_ADAPTABILITY",
    "offense",
    "role_adaptability",
    "major_strength",
    3,
    0.96,
    (
        "Scouting Report — passing and shot-selection evolution; "
        "Impact Evaluation — Cleveland, Miami, and return-to-Cleveland roles"
    ),
    (
        "LeBron altered his shot selection, passing, interior usage, "
        "on-ball burden, and complementary play as roster talent and age changed."
    ),
    (
        "The article predates the majority of his Lakers career."
    ),
)

add_lebron(
    "O_PORTABILITY",
    "offense",
    "offensive_portability",
    "strength",
    3,
    0.91,
    (
        "Key Stats and Trends and Impact Evaluation — "
        "offensive success across teams, roles, and lineup types"
    ),
    (
        "LeBron created historically strong offenses across multiple "
        "teams and retained value in a wider variety of roles than "
        "many other ball-dominant engines."
    ),
    (
        "Traditional interior fit and lack of elite spot-up shooting "
        "remain material limitations in some hypothetical roster constructions."
    ),
)


# -------------------------------------------------------------------
# LeBron James: defense
# -------------------------------------------------------------------

add_lebron(
    "D_POINT_OF_ATTACK",
    "defense",
    "point_of_attack",
    "major_strength",
    3,
    0.95,
    (
        "Scouting Report — perimeter-defense development "
        "and point, wing, and center matchup examples"
    ),
    (
        "At his defensive peak, LeBron combined quickness, strength, "
        "length, and technique to contain high-level perimeter creators."
    ),
    (
        "The source clearly distinguishes peak defense from weaker "
        "early seasons and reduced later-career activity."
    ),
)

add_lebron(
    "D_SCREEN_NAVIGATION",
    "defense",
    "screen_navigation",
    "strength",
    2,
    0.88,
    (
        "Scouting Report — strength through screens "
        "and pick-and-roll diagnosis"
    ),
    (
        "LeBron used strength, recognition, and anticipation to "
        "navigate or preempt screening actions during his peak."
    ),
    (
        "The article does not provide a complete screen-navigation "
        "success rate by season."
    ),
)

add_lebron(
    "D_HELP_POSITIONING",
    "defense",
    "help_positioning",
    "major_strength",
    3,
    0.97,
    (
        "Scouting Report — play diagnosis, help scoring, "
        "pinches, rotations, and pick-and-roll disruption"
    ),
    (
        "LeBron's peak help defense combined anticipation, positioning, "
        "size, and rapid rotations that could erase an opponent's advantage."
    ),
    (
        "The source also reports stationary off-ball possessions "
        "and missed rotations, especially outside his peak."
    ),
    sample_disclosed=True,
)

add_lebron(
    "D_ROTATION_TIMING",
    "defense",
    "rotation_timing",
    "mixed",
    2,
    0.92,
    (
        "Scouting Report — elite help sequences "
        "contrasted with bottom-quartile sampled rotation errors"
    ),
    (
        "LeBron produced exceptional rotations at his peak but also "
        "recorded meaningful missed-rotation and inactivity errors."
    ),
    (
        "The result is strongly career-phase dependent and should "
        "not be represented by a single fixed peak rating."
    ),
    sample_disclosed=True,
)

add_lebron(
    "D_PASSING_LANES",
    "defense",
    "passing_lane_disruption",
    "strength",
    3,
    0.92,
    (
        "Scouting Report — passing-lane jumps, "
        "pinches, help actions, and defensive anticipation"
    ),
    (
        "LeBron's size, anticipation, and court recognition enabled "
        "high-value passing-lane disruption from several positions."
    ),
    (
        "The source also identifies some risky gambles and therefore "
        "does not treat every steal attempt as positive value."
    ),
)

add_lebron(
    "D_RIM_PROTECTION",
    "defense",
    "rim_protection",
    "strength",
    3,
    0.94,
    (
        "Scouting Report — chase-down blocks, "
        "half-court rim protection, and block-location context"
    ),
    (
        "LeBron provided meaningful secondary rim protection, including "
        "high-value blocks near the basket and recovery plays in transition."
    ),
    (
        "He was not a full-time backline center, and the evidence "
        "should be interpreted as perimeter-player rim value."
    ),
)

add_lebron(
    "D_SWITCHABILITY",
    "defense",
    "switchability",
    "major_strength",
    3,
    0.98,
    (
        "Scouting Report — examples defending "
        "point, wing, and center positions"
    ),
    (
        "LeBron's peak combination of size, strength, speed, and "
        "technique gave him rare ability to survive switches across all positions."
    ),
    (
        "True five-position performance varied by matchup, season, "
        "and physical condition."
    ),
)

add_lebron(
    "D_POSITIONAL_VERSATILITY",
    "defense",
    "positional_versatility",
    "major_strength",
    3,
    0.98,
    (
        "Scouting Report — all-position coverage "
        "and Impact Evaluation defensive versatility"
    ),
    (
        "The source treats LeBron's ability to cover diverse roles "
        "and assignments as one of his defining defensive advantages."
    ),
    (
        "The evidence describes peak capability rather than identical "
        "versatility throughout every season."
    ),
)

add_lebron(
    "D_REBOUNDING",
    "defense",
    "defensive_rebounding",
    "major_strength",
    3,
    0.96,
    (
        "Scouting Report — long-term perimeter-player "
        "defensive-rebounding percentile"
    ),
    (
        "LeBron is evaluated as an elite defensive rebounder for a "
        "perimeter player from his teenage seasons onward."
    ),
    (
        "Position-relative percentile evidence does not fully account "
        "for lineup role or deliberate transition assignments."
    ),
    evidence_type=(
        "film_and_statistical_context"
    ),
)

add_lebron(
    "D_FOUL_DISCIPLINE",
    "defense",
    "foul_discipline",
    "major_strength",
    3,
    0.93,
    (
        "Scouting Report — shooting-foul frequency "
        "and physical wing defense"
    ),
    (
        "LeBron produced high-value defensive actions while committing "
        "unusually few shooting fouls."
    ),
    (
        "Low foul frequency does not independently establish complete "
        "defensive effectiveness or assignment difficulty."
    ),
    evidence_type=(
        "film_and_statistical_context"
    ),
)

add_lebron(
    "D_COMMUNICATION",
    "defense",
    "communication_recognition",
    "major_strength",
    3,
    0.95,
    (
        "Scouting Report — play diagnosis, "
        "mismatch recognition, and teammate direction"
    ),
    (
        "LeBron regularly anticipated opponent actions, recognized "
        "mismatches, and directed teammates before plays fully developed."
    ),
    (
        "Communication is difficult to observe consistently from "
        "broadcast footage and is supported by selected examples."
    ),
)

add_lebron(
    "D_GAMBLING_CONTROL",
    "defense",
    "gambling_error_control",
    "limitation",
    2,
    0.88,
    (
        "Scouting Report — Cleveland gambling, "
        "risky gambits, and rotation-error discussion"
    ),
    (
        "LeBron sometimes substituted risky anticipation plays for "
        "more reliable physical containment, creating avoidable breakdowns."
    ),
    (
        "The tendency varied by season and should not erase the "
        "large positive impact of his peak defensive activity."
    ),
    sample_disclosed=True,
)

add_lebron(
    "D_EFFORT_CONSISTENCY",
    "defense",
    "effort_consistency",
    "mixed",
    3,
    0.94,
    (
        "Scouting Report — 2009-13 peak, "
        "2014 activity decline, 2016 rebound, and later breakdowns"
    ),
    (
        "LeBron reached an elite activity level during his defensive "
        "peak but became substantially less consistent as mileage and "
        "offensive responsibility increased."
    ),
    (
        "Observed activity is affected by role, injury, workload, "
        "age, and deliberate energy allocation."
    ),
)

add_lebron(
    "D_PLAYOFF_ADAPTABILITY",
    "defense",
    "playoff_adaptability",
    "strength",
    3,
    0.92,
    (
        "Scouting Report and Impact Evaluation — "
        "multi-position assignments, playoff block peak, and key-stopper role"
    ),
    (
        "LeBron's peak versatility and recognition allowed him to "
        "change assignments and provide high-value defensive interventions "
        "in playoff settings."
    ),
    (
        "The source also documents reduced activity in later seasons "
        "and does not grade every postseason series."
    ),
)


def main() -> None:
    (
        sources,
        existing_claims,
        dimensions,
    ) = read_expert_evidence(
        settings.manual_dir
    )

    verifications = (
        read_source_verifications(
            settings.manual_dir
        )
    )

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
            "Missing Thinking Basketball "
            "sources: "
            f"{sorted(missing_sources)}"
        )

    imported = pd.DataFrame(
        claims
    )

    duplicate_claim_ids = imported[
        "CLAIM_ID"
    ][
        imported[
            "CLAIM_ID"
        ].duplicated(
            keep=False
        )
    ]

    if not duplicate_claim_ids.empty:
        raise ValueError(
            "Duplicate imported claim IDs: "
            f"{sorted(duplicate_claim_ids.unique())}"
        )

    retained = existing_claims[
        ~existing_claims[
            "SOURCE_ID"
        ].astype(str).isin(
            SOURCE_IDS
        )
    ].copy()

    combined = pd.concat(
        [
            retained,
            imported,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "SOURCE_ID",
            "PLAYER_NAME",
            "SIDE",
            "DIMENSION",
            "CLAIM_ID",
        ]
    ).reset_index(
        drop=True
    )

    validate_expert_evidence(
        sources,
        combined,
        dimensions,
    )

    validate_verified_claim_sources(
        combined,
        verifications,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        lineterminator="\n",
    )

    print(
        f"Wrote {len(imported)} "
        "Thinking Basketball claims."
    )

    print(
        f"Total expert claims: "
        f"{len(combined)}"
    )

    print()
    print(
        imported.groupby(
            [
                "PLAYER_NAME",
                "SIDE",
            ]
        )
        .size()
        .rename(
            "CLAIMS"
        )
    )


if __name__ == "__main__":
    main()
