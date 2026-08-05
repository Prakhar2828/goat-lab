from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import (
    confidence_badge,
    load_parquet,
    page_header,
    plain_language_intro,
    require_data,
)

page_header(
    "Skill profile and versatility",
    "Compare how each player created value instead of looking only at one final score.",
)
plain_language_intro(
    "Average league-relative performance across scoring, playmaking, rebounding, box-score defense, and team impact.",
    "Two players can reach similar overall value through very different strengths.",
    "Zero means roughly league average. Positive bars are above average; negative bars are below average.",
)

features = load_parquet("goat_player_season_values.parquet")
require_data(features)
regular = features[features["SEASON_TYPE"] == "Regular Season"]
family_columns = [column for column in regular.columns if column.startswith("FAMILY_")]
profile = regular.groupby("PLAYER_NAME")[family_columns].mean().reset_index()
long = profile.melt(
    id_vars="PLAYER_NAME",
    var_name="SKILL_FAMILY",
    value_name="Z_SCORE",
)
long["SKILL_FAMILY"] = (
    long["SKILL_FAMILY"]
    .str.replace("FAMILY_", "", regex=False)
    .str.replace("_", " ")
    .str.title()
)
figure = px.bar(
    long,
    x="SKILL_FAMILY",
    y="Z_SCORE",
    color="PLAYER_NAME",
    barmode="group",
    labels={
        "SKILL_FAMILY": "Skill family",
        "Z_SCORE": "League-relative score (0 = average)",
        "PLAYER_NAME": "Player",
    },
)
figure.add_hline(y=0)
st.plotly_chart(figure, use_container_width=True)

with st.expander("How to read a league-relative score"):
    st.markdown(
        """
- **0:** around league average for the season
- **+1:** about one standard deviation above league average
- **+2:** extremely far above league average
- **Negative:** below league average for that family

These values are adjusted within each season, so they compare dominance relative to the player's own league environment.
"""
    )

confidence_badge(
    "Mixed",
    "Scoring and playmaking have strong statistical coverage; full historical defense, off-ball value, and role versatility require more film evidence.",
)
