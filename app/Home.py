from __future__ import annotations

import plotly.express as px
import streamlit as st
import yaml

from common import load_parquet, metric_explainer, page_header

st.set_page_config(page_title="GOAT Lab", page_icon="🏀", layout="wide")

page_header(
    "GOAT Lab: Jordan vs. LeBron",
    "A coverage-aware, era-adjusted and assumption-transparent basketball greatness analysis.",
)

st.markdown(
    """
### The question is not only *who wins?*
It is **which player wins under which defensible definition of greatness, why, and how robust that conclusion is**.

Use the pages in the sidebar to inspect career value, peaks, playoffs, context, skill evidence,
cultural impact, individual arguments, and the interactive weighting model.
"""
)

scores = load_parquet("category_scores.parquet")
if not scores.empty:
    long = scores.melt(id_vars="PLAYER_NAME", var_name="CATEGORY", value_name="SCORE")
    figure = px.bar(
        long,
        x="CATEGORY",
        y="SCORE",
        color="PLAYER_NAME",
        barmode="group",
        range_y=[0, 100],
        title="Current category evidence scores",
    )
    st.plotly_chart(figure, use_container_width=True)
else:
    st.warning("Run `make all` to populate the analysis.")

with open("configs/weights.yaml", "r", encoding="utf-8") as handle:
    profiles = yaml.safe_load(handle)["profiles"]
st.subheader("Built-in philosophies")
st.dataframe(
    [dict(profile=name, **weights) for name, weights in profiles.items()],
    use_container_width=True,
    hide_index=True,
)
metric_explainer()
