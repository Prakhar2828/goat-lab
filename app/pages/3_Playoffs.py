from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import confidence_badge, load_parquet, page_header, require_data

page_header("Playoffs and opponent quality", "Production, opponent strength, expected series outcomes, and overperformance.")
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
    title="Postseason performance by year",
)
st.plotly_chart(figure, use_container_width=True)

series = load_parquet("playoff_series_scored.parquet")
if not series.empty:
    figure = px.scatter(
        series,
        x="EXPECTED_SERIES_WIN_PROB",
        y="SERIES_OVERPERFORMANCE",
        color="PLAYER_NAME" if "PLAYER_NAME" in series.columns else None,
        hover_data=[column for column in ["SEASON", "OPPONENT", "ROUND"] if column in series.columns],
        title="Actual series outcomes relative to pre-series expectation",
    )
    st.plotly_chart(figure, use_container_width=True)
else:
    st.warning("Add `data/manual/playoff_series.csv` and rerun `goatlab train-models` for contextual series analysis.")
confidence_badge(
    "Medium to high",
    "Box-score playoff evidence is broad; opponent and lineup context becomes less complete in earlier seasons.",
)
