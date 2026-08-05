from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import confidence_badge, load_parquet, page_header, require_data

page_header(
    "Team, teammate, and winning context",
    "Separate individual production from the conditions surrounding team outcomes.",
)

scores = load_parquet("production_category_scores.parquet")
series = load_parquet("playoff_series_scored.parquet")
require_data(scores)

context = scores[["PLAYER_NAME", "winning_context"]].copy()
context["CATEGORY"] = "Winning context"

figure = px.bar(
    context,
    x="PLAYER_NAME",
    y="winning_context",
    color="PLAYER_NAME",
    range_y=[0, 100],
    title="Frozen winning-context score",
)
st.plotly_chart(figure, use_container_width=True)

st.markdown(
    """
The winning-context category uses expected playoff-series outcomes to compare actual
team results with pre-series expectations. It is intentionally limited: the model does
not assign all team overperformance to one player.
"""
)

if not series.empty:
    columns = [
        column
        for column in [
            "PLAYER_NAME",
            "SEASON",
            "ROUND",
            "OPPONENT",
            "EXPECTED_SERIES_WIN_PROB",
            "SERIES_OVERPERFORMANCE",
            "TEAM_STAR_VALUE",
            "TEAM_SUPPORT_VALUE",
        ]
        if column in series.columns
    ]
    if columns:
        st.subheader("Playoff-series evidence")
        st.dataframe(
            series[columns],
            use_container_width=True,
            hide_index=True,
        )

confidence_badge(
    "Medium",
    "Opponent strength and series outcomes are modeled broadly, while coaching, "
    "injuries, role fit, and historical teammate detail remain only partially observable.",
)
