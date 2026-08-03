from __future__ import annotations

import pandas as pd
import streamlit as st

from common import page_header

page_header("Argument explorer", "The strongest version of every major argument, its counterargument, evidence, and verdict.")
path = "data/manual/arguments.csv"
try:
    arguments = pd.read_csv(path)
except FileNotFoundError:
    st.warning(f"Create `{path}` from the included template.")
    st.stop()

selected = st.selectbox("Argument", arguments["argument"].tolist())
row = arguments[arguments["argument"] == selected].iloc[0]
st.subheader(row["argument"])
st.markdown(f"**Usually supports:** {row['supports']}")
st.markdown(f"**Strongest evidence:** {row['supporting_evidence']}")
st.markdown(f"**Strongest counterargument:** {row['counterargument']}")
st.markdown(f"**Required analysis:** {row['analysis_required']}")
st.markdown(f"**Current verdict:** {row['verdict']}")
st.caption(f"Confidence: {row['confidence']} | Source IDs: {row['source_ids']}")
