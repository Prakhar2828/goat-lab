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
    "Playoffs and opponent quality",
    "Compare postseason performance and whether teams did better or worse than expected.",
)
plain_language_intro(
    "Playoff season value plus a pre-series model of how likely each team was to win.",
    "Rings and Finals records are team outcomes. This page adds opponent strength and pre-series expectations.",
    "Positive overperformance means a team exceeded the model's expectation; it does not assign all credit to one player.",
)

season_values = load_parquet("goat_player_season_values.parquet")
require_data(season_values)
playoffs = season_values[season_values["SEASON_TYPE"] == "Playoffs"]

figure = px.scatter(
    playoffs,
    x="SEASON",
    y="SEASON_VALUE_0_100",
    color="PLAYER_NAME",
    size="MIN",
    hover_data=["GP", "PTS_PER75", "TS_PCT", "AST_PER75"],
    labels={
        "SEASON": "Season",
        "SEASON_VALUE_0_100": "Playoff value (0–100)",
        "PLAYER_NAME": "Player",
        "MIN": "Minutes",
    },
    title="Postseason performance by year — larger circles mean more minutes",
)
st.plotly_chart(figure, use_container_width=True)

series = load_parquet("playoff_series_scored.parquet")
if not series.empty:
    st.subheader("Did the team beat the pre-series expectation?")
    figure = px.scatter(
        series,
        x="EXPECTED_SERIES_WIN_PROB",
        y="SERIES_OVERPERFORMANCE",
        color="PLAYER_NAME" if "PLAYER_NAME" in series.columns else None,
        hover_data=[
            column
            for column in ["SEASON", "OPPONENT", "ROUND"]
            if column in series.columns
        ],
        labels={
            "EXPECTED_SERIES_WIN_PROB": "Expected chance of winning before the series",
            "SERIES_OVERPERFORMANCE": "Actual result minus expected result",
            "PLAYER_NAME": "Player",
        },
    )
    figure.add_hline(y=0)
    st.plotly_chart(figure, use_container_width=True)
else:
    st.warning("Detailed playoff-series context is unavailable in this deployment.")

confidence_badge(
    "Medium to high",
    "Box-score playoff evidence is broad; opponent, injury, and lineup context becomes less complete in earlier seasons.",
)
