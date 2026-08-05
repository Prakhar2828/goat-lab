from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import load_parquet, page_header, plain_language_intro, require_data

page_header(
    "Peak, prime, and longevity",
    "Three different questions that are often incorrectly treated as one.",
)
plain_language_intro(
    "Several ways to measure how high a player reached, how long he stayed there, and how much career value he accumulated.",
    "Someone can have the best single season without having the best seven-year run or the most total career value.",
    "Compare the bars first, then open the detailed table for season counts, minutes, and career-value totals.",
)

summary = load_parquet("peak_prime_longevity.parquet")
require_data(summary)

window_labels = {
    "BEST_SEASON": "Best single season",
    "TOP_3_PEAK": "Best three seasons",
    "BEST_5_CONSECUTIVE": "Best five-year run",
    "BEST_7_CONSECUTIVE": "Best seven-year run",
    "TOP_10_SEASONS": "Best ten seasons",
}
shown = summary.melt(
    id_vars="PLAYER_NAME",
    value_vars=list(window_labels),
    var_name="WINDOW",
    value_name="VALUE",
)
shown["WINDOW"] = shown["WINDOW"].map(window_labels)
figure = px.bar(
    shown,
    x="WINDOW",
    y="VALUE",
    color="PLAYER_NAME",
    barmode="group",
    range_y=[0, 100],
    labels={
        "WINDOW": "Career window",
        "VALUE": "Season-value score (0–100)",
        "PLAYER_NAME": "Player",
    },
)
st.plotly_chart(figure, use_container_width=True)

with st.expander("See the full peak, prime, and longevity table"):
    renamed = summary.rename(
        columns={
            "PLAYER_NAME": "Player",
            "BEST_SEASON": "Best season",
            "TOP_3_PEAK": "Best three seasons",
            "BEST_3_CONSECUTIVE": "Best three-year run",
            "BEST_5_CONSECUTIVE": "Best five-year run",
            "BEST_7_CONSECUTIVE": "Best seven-year run",
            "TOP_10_SEASONS": "Best ten seasons",
            "ELITE_SEASONS": "Elite seasons",
            "ALL_NBA_LEVEL_SEASONS": "All-NBA-level seasons",
            "CAREER_VALUE_ABOVE_AVERAGE": "Career value above average",
            "CAREER_VALUE_ABOVE_ALL_STAR": "Career value above All-Star level",
            "SEASONS": "Seasons",
            "TOTAL_MINUTES": "Total minutes",
        }
    )
    st.dataframe(renamed, use_container_width=True, hide_index=True)

st.caption(
    "Consecutive windows break when there is a calendar-year gap, so retirement years are not silently treated as consecutive seasons."
)
