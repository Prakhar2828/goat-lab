from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    load_parquet,
    load_release_manifest,
    metric_explainer,
    page_header,
    require_data,
)

st.set_page_config(
    page_title="GOAT Lab v1",
    page_icon="🏀",
    layout="wide",
)

page_header(
    "GOAT Lab: Jordan vs. LeBron",
    "A frozen, reproducible comparison of basketball evidence and value assumptions.",
)

manifest = load_release_manifest()
scores = load_parquet("production_hierarchy_scores.parquet")
category_scores = load_parquet("production_category_scores.parquet")
simulation = load_parquet("weight_simulation_summary.parquet")

require_data(scores)
require_data(category_scores)
require_data(simulation)

central = manifest.get("central_result", {})
classification = manifest.get("result_classification", "conditional")
winner = str(central.get("winner", scores.iloc[0]["PLAYER_NAME"]))
margin = float(central.get("margin_points", 0.0))

win_rates = {
    str(row["PLAYER_NAME"]): float(row["WIN_RATE"])
    for _, row in simulation.iterrows()
}

st.markdown(
    """
### The result is close—and conditional

GOAT Lab does not claim to statistically prove a universal GOAT. It asks which player
ranks first under a preregistered production model, then shows which assumptions reverse
the result.
"""
)

metric_columns = st.columns(4)
metric_columns[0].metric("Frozen v1 leader", winner)
metric_columns[1].metric("Central margin", f"{margin:.6f} points")
metric_columns[2].metric(
    "LeBron simulation win rate",
    f"{win_rates.get('LeBron James', float('nan')):.4%}",
)
metric_columns[3].metric("Weight systems sampled", "250,000")

st.warning(
    "The winner is not stable across all approved scaling scenarios. "
    "The 60.1484% simulation result is the share of frozen-cap weight systems won, "
    "not an objective probability that LeBron is the GOAT."
)

st.subheader("Frozen production category scores")
long_scores = category_scores.melt(
    id_vars="PLAYER_NAME",
    var_name="CATEGORY",
    value_name="SCORE",
)
category_figure = px.bar(
    long_scores,
    x="CATEGORY",
    y="SCORE",
    color="PLAYER_NAME",
    barmode="group",
    range_y=[0, 100],
    title="Where each player gains ground in the production model",
)
st.plotly_chart(category_figure, use_container_width=True)

st.subheader("Approved scale sensitivity")
scale_rows = manifest.get("scale_sensitivity", {}).get("scenarios", [])
scale_frame = pd.DataFrame(scale_rows)
if not scale_frame.empty:
    shown = scale_frame.rename(
        columns={
            "scenario": "Scaling scenario",
            "lebron_score": "LeBron",
            "jordan_score": "Jordan",
            "lebron_minus_jordan": "LeBron − Jordan",
            "winner": "Winner",
        }
    )
    st.dataframe(
        shown[
            [
                "Scaling scenario",
                "LeBron",
                "Jordan",
                "LeBron − Jordan",
                "Winner",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    sensitivity_figure = px.bar(
        shown,
        x="Scaling scenario",
        y="LeBron − Jordan",
        color="Winner",
        title="Positive margins favor LeBron; negative margins favor Jordan",
    )
    sensitivity_figure.add_hline(y=0)
    st.plotly_chart(sensitivity_figure, use_container_width=True)

st.caption(
    "Release classification: "
    f"`{classification}`. Full assumptions, limitations, hashes, and the frozen source "
    "commit are documented in the repository."
)

metric_explainer()
