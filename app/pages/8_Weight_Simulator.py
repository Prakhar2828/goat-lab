from __future__ import annotations

import plotly.express as px
import streamlit as st
import yaml

from common import load_parquet, page_header, require_data
from goatlab.models.goat_score import score_players

page_header(
    "Build your definition of greatness",
    "Change the values, not the facts, and see where the conclusion changes.",
)
scores = load_parquet("category_scores.parquet")
require_data(scores)

with open("configs/weights.yaml", "r", encoding="utf-8") as handle:
    profiles = yaml.safe_load(handle)["profiles"]
profile_name = st.selectbox("Starting philosophy", list(profiles))
starting = profiles[profile_name]

weights: dict[str, float] = {}
columns = st.columns(3)
for index, category in enumerate(starting):
    with columns[index % 3]:
        weights[category] = st.slider(
            category.replace("_", " ").title(),
            0.0,
            1.0,
            float(starting[category]),
            0.01,
        )

ranked = score_players(scores, weights)
leader = ranked.iloc[0]
st.metric("Current leader", leader["PLAYER_NAME"], f"{leader['GOAT_SCORE']:.1f} / 100")
st.dataframe(
    ranked[["PLAYER_NAME", "GOAT_SCORE", "RANK"]],
    use_container_width=True,
    hide_index=True,
)

long = scores.melt(id_vars="PLAYER_NAME", var_name="CATEGORY", value_name="SCORE")
figure = px.bar(
    long,
    x="CATEGORY",
    y="SCORE",
    color="PLAYER_NAME",
    barmode="group",
    range_y=[0, 100],
    title="Evidence scores being weighted",
)
st.plotly_chart(figure, use_container_width=True)

simulation = load_parquet("weight_simulation_summary.parquet")
drivers = load_parquet("weight_simulation_drivers.parquet")
if not simulation.empty:
    st.subheader("Across 250,000 random reasonable definitions")
    simulation_display = simulation.copy()
    simulation_display["WIN_RATE"] = simulation_display["WIN_RATE"].map(lambda value: f"{value:.1%}")
    st.dataframe(simulation_display, use_container_width=True, hide_index=True)
if not drivers.empty:
    figure = px.bar(
        drivers,
        x="CATEGORY",
        y="MARGIN_CORRELATION_PLAYER_1",
        title="Which values most strongly move the result",
    )
    st.plotly_chart(figure, use_container_width=True)
