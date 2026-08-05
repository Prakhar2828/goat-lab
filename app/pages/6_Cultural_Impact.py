from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    confidence_badge,
    load_parquet,
    load_release_manifest,
    page_header,
    require_data,
)

page_header(
    "Cultural, commercial, and social impact",
    "A sourced legacy index with explicit sensitivity to the chosen subdimension weights.",
)

scores = load_parquet("production_category_scores.parquet")
manifest = load_release_manifest()
require_data(scores)

impact = scores[["PLAYER_NAME", "cultural_impact"]].copy()
figure = px.bar(
    impact,
    x="PLAYER_NAME",
    y="cultural_impact",
    color="PLAYER_NAME",
    range_y=[0, 100],
    title="Frozen cultural-impact score",
)
st.plotly_chart(figure, use_container_width=True)

sensitivity = manifest.get("cultural_weighting_sensitivity", {})
winner_counts = sensitivity.get("winner_counts", {})
if winner_counts:
    count_frame = pd.DataFrame(
        [
            {"PLAYER_NAME": player, "SCENARIOS_WON": count}
            for player, count in winner_counts.items()
        ]
    )
    st.subheader("Cultural weighting sensitivity")
    st.dataframe(count_frame, use_container_width=True, hide_index=True)

    columns = st.columns(2)
    columns[0].metric(
        "Weighting scenarios",
        int(sensitivity.get("scenario_count", count_frame["SCENARIOS_WON"].sum())),
    )
    columns[1].metric(
        "Attention-weight crossover",
        f"{float(sensitivity.get('configured_crossover_attention_weight', 0.0)):.3f}",
    )

st.markdown(
    """
The central cultural score is only one defensible weighting of attention, commercial
influence, philanthropy, and player influence. The ordering changes under reasonable
alternative weights, so this category is not presented as a settled fact.
"""
)

confidence_badge(
    "Low to medium",
    "Digital attention begins long after Jordan's playing peak, and commercial or "
    "philanthropic evidence is less standardized than basketball box-score data.",
)
