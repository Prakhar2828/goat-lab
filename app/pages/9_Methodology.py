from __future__ import annotations

from pathlib import Path

import streamlit as st
from common import metric_explainer, page_header

page_header(
    "How GOAT Lab works",
    "Start with the five-step explanation, then open any technical section you want.",
)

st.markdown(
    """
### The simple version

1. **Collect the evidence** — regular seasons, playoffs, awards, team context, and broader-impact records.
2. **Make eras more comparable** — adjust for pace, compare players with their own league, and reduce the influence of tiny samples.
3. **Build separate category scores** — peak, prime, longevity, regular season, playoffs, offense, defense, winning context, and cultural impact.
4. **Combine them with declared weights** — no hidden black-box final formula.
5. **Stress-test the conclusion** — test many alternative priorities and approved scoring methods to see what changes the winner.

The project is a transparent decision tool, not a machine that discovers one objective answer to a value-based debate.
"""
)

st.page_link(
    "pages/8_Weight_Simulator.py",
    label="🎛️ Try your own definition before reading the technical details",
    use_container_width=True,
)

metric_explainer()

st.header("Full project documentation")
documents = [
    ("Dashboard guide", "docs/DASHBOARD_GUIDE.md"),
    ("Final results", "docs/V1_FINAL_RESULTS.md"),
    ("What was frozen before the final run", "docs/V1_FINAL_PREREGISTRATION.md"),
    ("Complete methodology", "docs/METHODOLOGY.md"),
    ("Model card and risks", "docs/MODEL_CARD.md"),
    ("Dataset inventory and coverage", "docs/DATASETS.md"),
]

for title, document in documents:
    path = Path(document)
    if path.exists():
        with st.expander(title):
            st.markdown(path.read_text(encoding="utf-8"))
