from __future__ import annotations

import pandas as pd
import streamlit as st

from common import page_header, plain_language_intro

page_header(
    "Debate explorer",
    "See the strongest version of a popular argument, the best counterargument, and what v1 actually found.",
)
plain_language_intro(
    "Twenty common Jordan–LeBron arguments translated into testable questions.",
    "A memorable talking point can be true but still incomplete, selective, or unable to decide the full GOAT question.",
    "Choose an argument, then read the evidence, counterargument, v1 verdict, confidence, and the best page to explore next.",
)

path = "data/manual/arguments.csv"
try:
    arguments = pd.read_csv(path)
except FileNotFoundError:
    st.warning(f"Missing `{path}`.")
    st.stop()

support_filter = st.radio(
    "Show arguments usually used for",
    ["Both players", "Michael Jordan", "LeBron James"],
    horizontal=True,
)
filtered = arguments
if support_filter != "Both players":
    filtered = arguments[arguments["supports"] == support_filter]

selected = st.selectbox("Choose an argument", filtered["argument"].tolist())
row = filtered[filtered["argument"] == selected].iloc[0]

st.subheader(row["argument"])
left, right = st.columns(2)
left.markdown(f"**Usually supports:** {row['supports']}")
right.markdown(f"**Confidence in the v1 verdict:** {row['confidence']}")

st.markdown("### Why people believe it")
st.write(row["supporting_evidence"])

st.markdown("### Strongest counterargument")
st.write(row["counterargument"])

st.markdown("### What would be needed to test it well")
st.write(row["analysis_required"])

st.markdown("### GOAT Lab v1 verdict")
st.info(row["verdict"])

st.markdown(f"**Best page to explore next:** {row['relevant_page']}")

st.caption(
    "A verdict applies only to the evidence and definitions in v1. It is not a claim that the debate can never be interpreted differently."
)
