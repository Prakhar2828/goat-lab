from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    confidence_badge,
    load_parquet,
    load_release_manifest,
    page_header,
    plain_language_intro,
    require_data,
)

page_header(
    "Cultural and broader impact",
    "Compare influence beyond the box score while showing how uncertain that comparison is.",
)
plain_language_intro(
    "Public attention, commercial influence, social and philanthropic evidence, and influence on basketball culture.",
    "The GOAT debate often includes global reach and cultural impact, but those records are less standardized than basketball statistics.",
    "The central scores are close, and reasonable changes to the cultural subweights can reverse the order.",
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
    labels={
        "PLAYER_NAME": "Player",
        "cultural_impact": "Cultural-impact score (0–100)",
    },
)
st.plotly_chart(figure, use_container_width=True)

st.markdown(
    """
**Evidence considered**

- Public and media attention
- Commercial influence
- Social and philanthropic outcomes
- Influence on players and basketball culture
"""
)

sensitivity = manifest.get("cultural_weighting_sensitivity", {})
winner_counts = sensitivity.get("winner_counts", {})
if winner_counts:
    count_frame = pd.DataFrame(
        [
            {"Player": player, "Alternative cultural setups led": count}
            for player, count in winner_counts.items()
        ]
    )
    st.subheader("How often each player led under alternative cultural priorities")
    st.dataframe(count_frame, use_container_width=True, hide_index=True)

    columns = st.columns(2)
    columns[0].metric(
        "Cultural setups tested",
        int(sensitivity.get("scenario_count", count_frame.iloc[:, 1].sum())),
    )
    columns[1].metric(
        "Attention-weight crossover",
        f"{float(sensitivity.get('configured_crossover_attention_weight', 0.0)):.3f}",
        help="The approximate attention weight where the cultural leader changes.",
    )

st.info(
    "Digital attention data begins long after Jordan's playing peak. That makes this "
    "category less certain than career box-score evidence, so the dashboard presents "
    "sensitivity instead of pretending the ordering is settled."
)

confidence_badge(
    "Low to medium",
    "Digital attention favors modern eras, while older commercial and philanthropic evidence is less standardized and sometimes proprietary.",
)
