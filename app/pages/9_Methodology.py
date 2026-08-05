from __future__ import annotations

from pathlib import Path

import streamlit as st
from common import metric_explainer, page_header

page_header(
    "How GOAT Lab works",
    "Start with the simple explanation, then open the technical details that interest you.",
)

st.markdown(
    """
### The simple version

1. **Collect comparable evidence** from regular seasons, playoffs, defensive
   awards, team context, Wikimedia attention, and sourced cultural records.
2. **Make eras more comparable** by adjusting for pace, comparing each season
   with its own league, and reducing the influence of small samples.
3. **Build nine separate category scores** instead of hiding everything inside
   one black box.
4. **Combine those categories using weights frozen before the final result.**
5. **Stress-test the answer** using 250,000 alternative priority setups and four
   approved scaling methods.

GOAT Lab is a transparent comparison tool. It does not prove one objective GOAT.
"""
)

st.info(
    "No survey was conducted or used. No survey responses, external GOAT poll, "
    "RAPM model, PCA-derived category score, or gradient-boosted playoff model "
    "entered the published v1 result."
)

st.page_link(
    "pages/8_Weight_Simulator.py",
    label="🎛️ Build your own GOAT definition",
    use_container_width=True,
)

metric_explainer()

st.header("What entered the published result")

entered_col, excluded_col = st.columns(2)

with entered_col:
    st.markdown(
        """
#### Included

- Era-relative player-season statistics
- Transparent five-family season value
- Historical career reference distributions
- Regularized playoff-series logistic regression
- Defensive box and award evidence
- Wikimedia attention
- Sourced cultural-impact rubric
- Frozen category weights
- 250,000 hierarchy-aware weight draws
"""
    )

with excluded_col:
    st.markdown(
        """
#### Not included

- Surveys or external GOAT polls
- RAPM or possession-level lineup models
- Shot-chart or play-type models
- PCA-derived category scores
- Bootstrap confidence intervals
- Gradient-boosted playoff models
- Expert-film scores that failed eligibility
- Google Trends, GDELT, or New York Times inputs
"""
    )

st.header("How to interpret the result")

st.markdown(
    """
The published setup gives LeBron a score of **89.258985** and Jordan a score of
**89.143895**.

The difference is **0.115091 points on the 100-point display scale**.

In the weight stress test, LeBron ranked first in **60.1484%** of the sampled
setups and Jordan ranked first in **39.8516%**.

Those percentages describe the scoring setups tested. They are not objective
probabilities that either player is the GOAT.

The four approved scaling methods split 2–2 between the players, so the final
conclusion is intentionally described as conditional and assumption-sensitive.
"""
)

st.header("Published v1 documentation")

documents = [
    ("Complete executed methodology", "docs/METHODOLOGY.md"),
    ("Dataset inventory actually used", "docs/DATASETS.md"),
    ("Model card and risks", "docs/MODEL_CARD.md"),
    ("Dashboard guide", "docs/DASHBOARD_GUIDE.md"),
    ("Final results", "docs/V1_FINAL_RESULTS.md"),
    (
        "What was frozen before the final run",
        "docs/V1_FINAL_PREREGISTRATION.md",
    ),
]

for title, document in documents:
    path = Path(document)

    if path.exists():
        with st.expander(title):
            st.markdown(path.read_text(encoding="utf-8"))