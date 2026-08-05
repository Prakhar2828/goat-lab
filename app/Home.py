from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    friendly_category,
    load_parquet,
    load_release_manifest,
    metric_explainer,
    require_data,
)

st.set_page_config(
    page_title="GOAT Lab",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 GOAT Lab")
st.subheader("Michael Jordan vs. LeBron James — with the assumptions made visible")

st.markdown(
    """
GOAT Lab is an interactive comparison of **peak, prime, longevity, regular-season play,
playoffs, offense, defense, winning context, and cultural impact**.

It is **not** an attempt to prove one unquestionable GOAT. It shows which player leads
under one declared scoring setup, then lets you see which reasonable assumptions change
the answer.
"""
)

manifest = load_release_manifest()
scores = load_parquet("production_hierarchy_scores.parquet")
category_scores = load_parquet("production_category_scores.parquet")
simulation = load_parquet("weight_simulation_summary.parquet")

require_data(scores)
require_data(category_scores)
require_data(simulation)

central = manifest.get("central_result", {})
winner = str(central.get("winner", scores.iloc[0]["PLAYER_NAME"]))
margin = float(central.get("margin_points", 0.0))
score_lookup = {
    str(row["PLAYER_NAME"]): float(row["GOAT_SCORE"])
    for _, row in scores.iterrows()
}
win_rates = {
    str(row["PLAYER_NAME"]): float(row["WIN_RATE"])
    for _, row in simulation.iterrows()
}
lebron_rate = win_rates.get("LeBron James", float("nan"))

st.divider()
st.header("My default scoring setup")

left, middle, right = st.columns(3)
left.metric("Leader", winner)
middle.metric("Score difference", f"{margin:.3f} points")
right.metric("LeBron-led stress-test setups", f"{lebron_rate:.1%}")

st.markdown(
    f"""
**LeBron James: {score_lookup.get('LeBron James', float('nan')):.3f}**  
**Michael Jordan: {score_lookup.get('Michael Jordan', float('nan')):.3f}**

The score difference is only **{margin:.3f} points on a 100-point scale**, so the default
result is extremely close.
"""
)

st.info(
    "LeBron leads narrowly in my published setup. In the stress test, LeBron led in "
    f"{lebron_rate:.1%} of the alternative setups and Jordan led in "
    f"{1.0 - lebron_rate:.1%}. That percentage describes the setups tested—not an "
    "objective probability that either player is the GOAT."
)

st.header("Think the model values the wrong things?")
st.write(
    "Change the importance of peak, longevity, offense, defense, playoffs, and winning "
    "context. The simulator immediately shows which player your definition favors."
)
st.page_link(
    "pages/8_Weight_Simulator.py",
    label="🎛️ Build your own GOAT definition",
    use_container_width=True,
)

with st.expander("How the 250,000-setup stress test worked"):
    st.markdown(
        """
The technical name is a **Monte Carlo simulation**. GOAT Lab repeatedly sampled
alternative priorities while keeping the overall structure fixed:

- **50% Career performance:** peak, prime, longevity, regular season, and playoffs
- **40% Basketball value:** offense, defense, and winning context
- **10% Broader influence:** cultural impact

Only the priorities **inside** the first two sections changed. The three section totals
always remained 50% / 40% / 10%. The published run used 250,000 samples, random seed 23,
and a within-section concentration of 100.

**My result** is the single published default setup.  
**The stress-test result** asks how often each player leads when those internal priorities
are changed in many reasonable ways.
"""
    )

st.divider()
st.header("Where each player gains ground")
st.write(
    "These are the frozen evidence scores that the default setup and the simulator both "
    "weight. Higher means stronger evidence within that category."
)

long_scores = category_scores.melt(
    id_vars="PLAYER_NAME",
    var_name="CATEGORY",
    value_name="SCORE",
)
long_scores["CATEGORY"] = long_scores["CATEGORY"].map(friendly_category)
category_figure = px.bar(
    long_scores,
    x="CATEGORY",
    y="SCORE",
    color="PLAYER_NAME",
    barmode="group",
    range_y=[0, 100],
    labels={
        "CATEGORY": "Category",
        "SCORE": "Evidence score (0–100)",
        "PLAYER_NAME": "Player",
    },
)
st.plotly_chart(category_figure, use_container_width=True)

st.caption(
    "Largest visible split: Jordan gains strongly from defense; LeBron gains from "
    "offense, longevity, and several performance-arc categories."
)

with st.expander("Why another reasonable scoring method can change the winner"):
    scale_rows = manifest.get("scale_sensitivity", {}).get("scenarios", [])
    scale_frame = pd.DataFrame(scale_rows)
    if not scale_frame.empty:
        shown = scale_frame.rename(
            columns={
                "scenario": "Scoring method",
                "lebron_score": "LeBron",
                "jordan_score": "Jordan",
                "lebron_minus_jordan": "Score difference (LeBron − Jordan)",
                "winner": "Leader",
            }
        )
        st.write(
            "Four approved ways of placing elite career values on a 0–100 scale were "
            "tested. Two favor LeBron and two favor Jordan, so the conclusion is "
            "conditional rather than universal."
        )
        st.dataframe(
            shown[
                [
                    "Scoring method",
                    "LeBron",
                    "Jordan",
                    "Score difference (LeBron − Jordan)",
                    "Leader",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

metric_explainer()
