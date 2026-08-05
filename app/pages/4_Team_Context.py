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
    "Team and winning context",
    "Separate individual performance from the situation surrounding team results.",
)
plain_language_intro(
    "How often each player's teams performed above or below a model's pre-series expectation.",
    "Championships matter, but they are produced by players, teammates, coaching, injuries, opponents, and many other conditions.",
    "Treat this as context—not as a claim that one player caused every team result.",
)

scores = load_parquet("production_category_scores.parquet")
series = load_parquet("playoff_series_scored.parquet")
require_data(scores)

context = scores[["PLAYER_NAME", "winning_context"]].copy()
figure = px.bar(
    context,
    x="PLAYER_NAME",
    y="winning_context",
    color="PLAYER_NAME",
    range_y=[0, 100],
    labels={
        "PLAYER_NAME": "Player",
        "winning_context": "Winning-context score (0–100)",
    },
    title="Frozen winning-context score",
)
st.plotly_chart(figure, use_container_width=True)

st.write(
    "The score begins with expected playoff-series outcomes and compares them with what "
    "actually happened. It intentionally does not assign all team overperformance to the "
    "featured player."
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
        with st.expander("Explore the playoff-series evidence"):
            renamed = series[columns].rename(
                columns={
                    "PLAYER_NAME": "Player",
                    "SEASON": "Season",
                    "ROUND": "Round",
                    "OPPONENT": "Opponent",
                    "EXPECTED_SERIES_WIN_PROB": "Expected win probability",
                    "SERIES_OVERPERFORMANCE": "Overperformance",
                    "TEAM_STAR_VALUE": "Star value",
                    "TEAM_SUPPORT_VALUE": "Supporting-cast value",
                }
            )
            st.dataframe(renamed, use_container_width=True, hide_index=True)

confidence_badge(
    "Medium",
    "Opponent strength and series outcomes are modeled broadly, while coaching, injuries, role fit, and historical teammate detail remain only partially observable.",
)
