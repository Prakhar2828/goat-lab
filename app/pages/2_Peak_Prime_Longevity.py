from __future__ import annotations

import plotly.express as px
from common import load_parquet, page_header, require_data

page_header("Peak, prime, and longevity", "Separate answers to three questions that are often incorrectly combined.")
summary = load_parquet("peak_prime_longevity.parquet")
require_data(summary)

shown = summary.melt(
    id_vars="PLAYER_NAME",
    value_vars=[
        "BEST_SEASON",
        "TOP_3_PEAK",
        "BEST_5_CONSECUTIVE",
        "BEST_7_CONSECUTIVE",
        "TOP_10_SEASONS",
    ],
    var_name="WINDOW",
    value_name="VALUE",
)
figure = px.bar(shown, x="WINDOW", y="VALUE", color="PLAYER_NAME", barmode="group")
figure.update_yaxes(range=[0, 100])
import streamlit as st

st.plotly_chart(figure, use_container_width=True)
st.dataframe(summary, use_container_width=True, hide_index=True)
