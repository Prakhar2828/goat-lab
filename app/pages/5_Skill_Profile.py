from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import confidence_badge, load_parquet, page_header, require_data

page_header("Skill profile and versatility", "Compare how value was created rather than collapsing everything into one total.")
features = load_parquet("goat_player_season_values.parquet")
require_data(features)
regular = features[features["SEASON_TYPE"] == "Regular Season"]
family_columns = [column for column in regular.columns if column.startswith("FAMILY_")]
profile = regular.groupby("PLAYER_NAME")[family_columns].mean().reset_index()
long = profile.melt(id_vars="PLAYER_NAME", var_name="SKILL_FAMILY", value_name="Z_SCORE")
figure = px.bar(long, x="SKILL_FAMILY", y="Z_SCORE", color="PLAYER_NAME", barmode="group")
st.plotly_chart(figure, use_container_width=True)

confidence_badge(
    "Mixed",
    "Scoring and playmaking have strong statistical coverage; historical defense and role versatility require film evidence.",
)
