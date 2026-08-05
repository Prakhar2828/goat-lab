from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import load_parquet, page_header, plain_language_intro, require_data

page_header(
    "Career curves",
    "See how each player's value rose, peaked, lasted, and changed over time.",
)
plain_language_intro(
    "One point for every regular season or playoff run, ordered by career year.",
    "A GOAT case can be built on a higher peak, a longer prime, or more accumulated value.",
    "Use the Regular Season / Playoffs switch, then hover over any point for the exact season, games, and minutes.",
)

features = load_parquet("goat_player_season_values.parquet")
require_data(features)

season_type = st.radio(
    "Choose a part of the season",
    ["Regular Season", "Playoffs"],
    horizontal=True,
)
view = features[features["SEASON_TYPE"] == season_type].copy()

best_rows = (
    view.sort_values("SEASON_VALUE_0_100", ascending=False)
    .groupby("PLAYER_NAME", as_index=False)
    .first()
)
columns = st.columns(len(best_rows))
for column, (_, row) in zip(columns, best_rows.iterrows(), strict=False):
    column.metric(
        f"Best {season_type.lower()} value — {row['PLAYER_NAME']}",
        f"{float(row['SEASON_VALUE_0_100']):.1f}",
        help=f"Season: {row['SEASON']}",
    )

figure = px.line(
    view,
    x="CAREER_YEAR",
    y="SEASON_VALUE_0_100",
    color="PLAYER_NAME",
    markers=True,
    hover_data=["SEASON", "GP", "MIN"],
    labels={
        "CAREER_YEAR": "Career year",
        "SEASON_VALUE_0_100": "Season value (0–100)",
        "PLAYER_NAME": "Player",
    },
    title=f"How each player's {season_type.lower()} value changed by career year",
)
figure.update_yaxes(range=[0, 100])
st.plotly_chart(figure, use_container_width=True)

st.subheader("Accumulated value above an average season")
st.write(
    "Each season adds only the amount above 50. A season at or below 50 adds zero rather "
    "than subtracting from the career total."
)
view = view.sort_values(["PLAYER_NAME", "CAREER_YEAR"])
view["CUMULATIVE_VALUE_ABOVE_AVERAGE"] = view.groupby("PLAYER_NAME")[
    "SEASON_VALUE_0_100"
].transform(lambda values: (values - 50).clip(lower=0).cumsum())
figure = px.line(
    view,
    x="CAREER_YEAR",
    y="CUMULATIVE_VALUE_ABOVE_AVERAGE",
    color="PLAYER_NAME",
    markers=True,
    labels={
        "CAREER_YEAR": "Career year",
        "CUMULATIVE_VALUE_ABOVE_AVERAGE": "Accumulated value above average",
        "PLAYER_NAME": "Player",
    },
)
st.plotly_chart(figure, use_container_width=True)
