from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import confidence_badge, load_parquet, page_header

page_header("Team, teammate, and winning context", "Distinguish individual value from roster quality and team outcomes.")
series = load_parquet("playoff_series_scored.parquet")
team_srs = load_parquet("team_srs.parquet")

if not team_srs.empty:
    figure = px.line(team_srs, x="SEASON", y="SRS_EST", color="TEAM_ID", title="Estimated team strength by season")
    st.plotly_chart(figure, use_container_width=True)
else:
    st.warning("Team SRS data has not been built.")

if not series.empty:
    columns = [
        column
        for column in [
            "PLAYER_NAME",
            "SEASON",
            "ROUND",
            "OPPONENT",
            "EXPECTED_SERIES_WIN_PROB",
            "SERIES_OVERPERFORMANCE",
            "TEAM_STAR_VALUE",
            "TEAM_SUPPORT_VALUE",
        ]
        if column in series.columns
    ]
    st.dataframe(series[columns], use_container_width=True, hide_index=True)

confidence_badge(
    "Medium",
    "Team context can be estimated, but coaching, injuries, role fit, and roster changes are only partially observable.",
)
