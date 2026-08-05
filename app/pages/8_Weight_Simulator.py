from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    load_parquet,
    load_release_manifest,
    page_header,
    require_data,
)

from goatlab.models.goat_score import score_players

page_header(
    "Build your definition of greatness",
    "Adjust emphasis within the frozen 50% / 40% / 10% hierarchy and see where the result changes.",
)

scores = load_parquet("production_category_scores.parquet")
simulation = load_parquet("weight_simulation_summary.parquet")
drivers = load_parquet("weight_simulation_drivers.parquet")
manifest = load_release_manifest()

require_data(scores)
require_data(drivers)

st.info(
    "This interactive view is exploratory. The published v1 result remains the "
    "preregistered central configuration and is not changed by these sliders."
)

group_caps = {
    str(row["GROUP"]): float(row["FROZEN_GROUP_CAP"])
    for row in manifest.get("group_cap_audit", [])
}

ordered_groups = [
    group
    for group in ("performance_arc", "basketball_value", "broader_legacy")
    if group in set(drivers["GROUP"])
]

weights: dict[str, float] = {}
weight_rows: list[dict[str, float | str]] = []

for group in ordered_groups:
    group_frame = drivers[drivers["GROUP"] == group].copy()
    group_frame = group_frame.sort_values(
        "FROZEN_TOTAL_WEIGHT",
        ascending=False,
    )
    cap = group_caps.get(
        group,
        float(group_frame["FROZEN_TOTAL_WEIGHT"].sum()),
    )

    st.subheader(
        f"{group.replace('_', ' ').title()} — fixed group mass {cap:.0%}"
    )

    if len(group_frame) == 1:
        category = str(group_frame.iloc[0]["CATEGORY"])
        weights[category] = cap
        st.caption(f"{category.replace('_', ' ').title()}: fixed at {cap:.0%}")
        weight_rows.append(
            {
                "GROUP": group,
                "CATEGORY": category,
                "EFFECTIVE_WEIGHT": cap,
            }
        )
        continue

    raw_values: dict[str, float] = {}
    slider_columns = st.columns(min(3, len(group_frame)))

    for index, (_, row) in enumerate(group_frame.iterrows()):
        category = str(row["CATEGORY"])
        frozen_total = float(row["FROZEN_TOTAL_WEIGHT"])
        default_share = frozen_total / cap

        with slider_columns[index % len(slider_columns)]:
            raw_values[category] = st.slider(
                category.replace("_", " ").title(),
                min_value=0.01,
                max_value=1.00,
                value=float(round(default_share, 2)),
                step=0.01,
                key=f"{group}_{category}",
            )

    raw_total = sum(raw_values.values())
    for category, raw_value in raw_values.items():
        effective = cap * raw_value / raw_total
        weights[category] = effective
        weight_rows.append(
            {
                "GROUP": group,
                "CATEGORY": category,
                "EFFECTIVE_WEIGHT": effective,
            }
        )

ranked = score_players(scores, weights)
leader = ranked.iloc[0]
runner_up = ranked.iloc[1]
margin = float(leader["GOAT_SCORE"] - runner_up["GOAT_SCORE"])

metric_columns = st.columns(3)
metric_columns[0].metric("Current leader", str(leader["PLAYER_NAME"]))
metric_columns[1].metric("Current score", f"{float(leader['GOAT_SCORE']):.3f}")
metric_columns[2].metric("Current margin", f"{margin:.3f}")

st.dataframe(
    ranked[["PLAYER_NAME", "GOAT_SCORE", "RANK"]],
    use_container_width=True,
    hide_index=True,
)

weight_frame = pd.DataFrame(weight_rows)
st.subheader("Effective normalized weights")
st.dataframe(
    weight_frame.assign(
        EFFECTIVE_WEIGHT=lambda frame: frame["EFFECTIVE_WEIGHT"].map(
            lambda value: f"{value:.2%}"
        )
    ),
    use_container_width=True,
    hide_index=True,
)

long_scores = scores.melt(
    id_vars="PLAYER_NAME",
    var_name="CATEGORY",
    value_name="SCORE",
)
figure = px.bar(
    long_scores,
    x="CATEGORY",
    y="SCORE",
    color="PLAYER_NAME",
    barmode="group",
    range_y=[0, 100],
    title="Frozen evidence scores being weighted",
)
st.plotly_chart(figure, use_container_width=True)

if not simulation.empty:
    st.subheader("Preregistered 250,000-run simulation")
    simulation_display = simulation.copy()
    simulation_display["WIN_RATE"] = simulation_display["WIN_RATE"].map(
        lambda value: f"{value:.4%}"
    )
    st.dataframe(
        simulation_display,
        use_container_width=True,
        hide_index=True,
    )

driver_figure = px.bar(
    drivers.sort_values("MARGIN_CORRELATION_PLAYER_1"),
    x="CATEGORY",
    y="MARGIN_CORRELATION_PLAYER_1",
    color="GROUP",
    title="Which within-group weight changes most strongly move the LeBron − Jordan margin",
)
driver_figure.add_hline(y=0)
st.plotly_chart(driver_figure, use_container_width=True)
