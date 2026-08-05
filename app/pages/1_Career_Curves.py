from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import load_parquet, page_header, require_data

page_header("Career curves", "Peak height, sustained prime, decline, and accumulated career value.")
features = load_parquet("goat_player_season_values.parquet")
require_data(features)

season_type = st.radio("Season segment", ["Regular Season", "Playoffs"], horizontal=True)
view = features[features["SEASON_TYPE"] == season_type].copy()

figure = px.line(
    view,
    x="CAREER_YEAR",
    y="SEASON_VALUE_0_100",
    color="PLAYER_NAME",
    markers=True,
    hover_data=["SEASON", "GP", "MIN"],
    title=f"Season value by career year — {season_type}",
)
figure.update_yaxes(range=[0, 100])
st.plotly_chart(figure, use_container_width=True)

view = view.sort_values(["PLAYER_NAME", "CAREER_YEAR"])
view["CUMULATIVE_VALUE_ABOVE_AVERAGE"] = view.groupby("PLAYER_NAME")["SEASON_VALUE_0_100"].transform(
    lambda values: (values - 50).clip(lower=0).cumsum()
)
figure = px.line(
    view,
    x="CAREER_YEAR",
    y="CUMULATIVE_VALUE_ABOVE_AVERAGE",
    color="PLAYER_NAME",
    markers=True,
    title="Accumulated value above an average NBA season",
)
st.plotly_chart(figure, use_container_width=True)
