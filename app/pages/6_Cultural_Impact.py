from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import confidence_badge, load_parquet, page_header

page_header("Cultural, commercial, and social impact", "Separate attention, business influence, philanthropy, and player influence.")
pageviews = load_parquet("wikimedia_pageviews.parquet")
if not pageviews.empty:
    monthly = (
        pageviews.assign(month=lambda frame: frame["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["PLAYER_NAME", "month"], as_index=False)["views"]
        .sum()
    )
    figure = px.line(monthly, x="month", y="views", color="PLAYER_NAME", title="Wikipedia attention by month")
    st.plotly_chart(figure, use_container_width=True)
else:
    st.warning("Run `goatlab ingest-cultural` to fetch Wikimedia attention data.")

st.subheader("Verified impact ledger")
st.write(
    "Populate `data/manual/impact_ledger.csv` with one sourced event per row: commercial milestone, "
    "verified donation, institution created, beneficiary count, media milestone, or documented influence statement."
)
confidence_badge(
    "Low to medium",
    "Digital attention begins long after Jordan's playing peak, so modern web signals are shown in overlapping windows and never treated as full-career equivalence.",
)
