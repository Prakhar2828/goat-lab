from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    friendly_category,
    load_parquet,
    load_release_manifest,
    page_header,
    plain_language_intro,
    require_data,
)

from goatlab.models.goat_score import score_players

page_header(
    "Build your own GOAT definition",
    "Change what matters to you and see which player your priorities favor.",
)
plain_language_intro(
    "Your priorities inside three fixed sections: career performance, basketball value, and broader influence.",
    "The debate changes when people value peak, longevity, offense, defense, playoffs, or winning differently.",
    "Move the priority sliders or apply a preset. GOAT Lab converts them into percentages that always total 100%.",
)

scores = load_parquet("production_category_scores.parquet")
simulation = load_parquet("weight_simulation_summary.parquet")
drivers = load_parquet("weight_simulation_drivers.parquet")
manifest = load_release_manifest()

require_data(scores)
require_data(drivers)

st.info(
    "This is your interactive result. It does not change the published default result or rerun the 250,000-setup stress test."
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

default_shares: dict[str, float] = {}
frozen_weights: dict[str, float] = {}
for _, row in drivers.iterrows():
    category = str(row["CATEGORY"])
    group = str(row["GROUP"])
    cap = group_caps[group]
    total_weight = float(row["FROZEN_TOTAL_WEIGHT"])
    frozen_weights[category] = total_weight
    default_shares[category] = total_weight / cap

presets: dict[str, dict[str, float]] = {
    "Published default": default_shares,
    "Peak matters most": {
        **default_shares,
        "peak": 0.45,
        "prime": 0.20,
        "longevity": 0.05,
        "regular_season": 0.15,
        "playoffs": 0.15,
    },
    "Longevity matters most": {
        **default_shares,
        "peak": 0.10,
        "prime": 0.15,
        "longevity": 0.45,
        "regular_season": 0.15,
        "playoffs": 0.15,
    },
    "Two-way dominance": {
        **default_shares,
        "offense": 0.35,
        "defense": 0.50,
        "winning_context": 0.15,
    },
    "Offensive engine": {
        **default_shares,
        "offense": 0.65,
        "defense": 0.20,
        "winning_context": 0.15,
    },
    "Championship context": {
        **default_shares,
        "offense": 0.30,
        "defense": 0.20,
        "winning_context": 0.50,
    },
}

preset_columns = st.columns([3, 1])
selected_preset = preset_columns[0].selectbox(
    "Try a preset philosophy",
    list(presets),
)
if preset_columns[1].button("Apply preset", use_container_width=True):
    for category, value in presets[selected_preset].items():
        st.session_state[f"priority_{category}"] = float(value)
    st.rerun()

st.caption(
    "The slider numbers are priorities inside each section. They are automatically normalized, so you do not need to make them add to 1.0 yourself."
)

weights: dict[str, float] = {}
weight_rows: list[dict[str, float | str]] = []
section_names = {
    "performance_arc": "Career performance",
    "basketball_value": "Basketball value",
    "broader_legacy": "Broader influence",
}

for group in ordered_groups:
    group_frame = drivers[drivers["GROUP"] == group].copy()
    group_frame = group_frame.sort_values("FROZEN_TOTAL_WEIGHT", ascending=False)
    cap = group_caps.get(group, float(group_frame["FROZEN_TOTAL_WEIGHT"].sum()))

    st.subheader(f"{section_names[group]} — fixed at {cap:.0%} of the total")

    if len(group_frame) == 1:
        category = str(group_frame.iloc[0]["CATEGORY"])
        weights[category] = cap
        st.caption(f"{friendly_category(category)} stays fixed at {cap:.0%}.")
        weight_rows.append(
            {
                "Section": section_names[group],
                "Category": friendly_category(category),
                "Effective weight": cap,
            }
        )
        continue

    raw_values: dict[str, float] = {}
    slider_columns = st.columns(min(3, len(group_frame)))

    for index, (_, row) in enumerate(group_frame.iterrows()):
        category = str(row["CATEGORY"])
        key = f"priority_{category}"
        if key not in st.session_state:
            st.session_state[key] = float(round(default_shares[category], 2))
        with slider_columns[index % len(slider_columns)]:
            raw_values[category] = st.slider(
                f"{friendly_category(category)} priority",
                min_value=0.01,
                max_value=1.00,
                step=0.01,
                key=key,
                help="A larger value gives this category more of its section's fixed total.",
            )

    raw_total = sum(raw_values.values())
    for category, raw_value in raw_values.items():
        effective = cap * raw_value / raw_total
        weights[category] = effective
        weight_rows.append(
            {
                "Section": section_names[group],
                "Category": friendly_category(category),
                "Effective weight": effective,
            }
        )

total_weight = sum(weights.values())
summary_columns = st.columns(4)
summary_columns[0].metric("Total model weight", f"{total_weight:.1%}")
for index, group in enumerate(ordered_groups, start=1):
    section_total = sum(
        weight
        for category, weight in weights.items()
        if str(drivers.loc[drivers["CATEGORY"] == category, "GROUP"].iloc[0]) == group
    )
    summary_columns[index].metric(section_names[group], f"{section_total:.1%}")

if total_weight > 1.000001:
    st.error("Your effective weights exceed 100%. Reduce one or more priorities.")
elif total_weight < 0.999999:
    st.warning("Your effective weights are below 100%. Increase one or more priorities.")
else:
    st.success("Your effective weights total exactly 100%. The app normalized them automatically.")

ranked = score_players(scores, weights)
leader = ranked.iloc[0]
runner_up = ranked.iloc[1]
margin = float(leader["GOAT_SCORE"] - runner_up["GOAT_SCORE"])

st.divider()
st.header(f"Your definition favors {leader['PLAYER_NAME']}")
metric_columns = st.columns(3)
metric_columns[0].metric("Leader", str(leader["PLAYER_NAME"]))
metric_columns[1].metric("Leader's score", f"{float(leader['GOAT_SCORE']):.3f}")
metric_columns[2].metric("Score difference", f"{margin:.3f}")

correlations = {
    str(row["CATEGORY"]): float(row["MARGIN_CORRELATION_PLAYER_1"])
    for _, row in drivers.iterrows()
}
movements = []
for category, effective in weights.items():
    change = effective - frozen_weights.get(category, effective)
    movement = change * correlations.get(category, 0.0)
    movements.append((abs(movement), movement, category))
movements.sort(reverse=True)
reasons = []
for _, movement, category in movements[:2]:
    if abs(movement) < 1e-6:
        continue
    direction = "toward LeBron" if movement > 0 else "toward Jordan"
    reasons.append(f"{friendly_category(category)} moved the result {direction}")
if reasons:
    st.write("The biggest changes from the published default were: " + "; ".join(reasons) + ".")
else:
    st.write("Your priorities are very close to the published default setup.")

st.dataframe(
    ranked[["PLAYER_NAME", "GOAT_SCORE", "RANK"]].rename(
        columns={
            "PLAYER_NAME": "Player",
            "GOAT_SCORE": "GOAT score",
            "RANK": "Rank",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

with st.expander("See the exact normalized weights"):
    weight_frame = pd.DataFrame(weight_rows)
    st.dataframe(
        weight_frame.assign(
            **{
                "Effective weight": weight_frame["Effective weight"].map(
                    lambda value: f"{value:.2%}"
                )
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("See the frozen evidence scores being weighted"):
    long_scores = scores.melt(
        id_vars="PLAYER_NAME",
        var_name="CATEGORY",
        value_name="SCORE",
    )
    long_scores["CATEGORY"] = long_scores["CATEGORY"].map(friendly_category)
    figure = px.bar(
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
    st.plotly_chart(figure, use_container_width=True)

with st.expander("What the published 250,000-setup stress test found"):
    if not simulation.empty:
        simulation_display = simulation.copy()
        simulation_display["WIN_RATE"] = simulation_display["WIN_RATE"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(
            simulation_display.rename(
                columns={
                    "PLAYER_NAME": "Player",
                    "WIN_RATE": "Setups led",
                    "MEAN_SCORE": "Mean score",
                    "P05_SCORE": "5th percentile score",
                    "P95_SCORE": "95th percentile score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    driver_figure = px.bar(
        drivers.assign(CATEGORY=drivers["CATEGORY"].map(friendly_category)).sort_values(
            "MARGIN_CORRELATION_PLAYER_1"
        ),
        x="CATEGORY",
        y="MARGIN_CORRELATION_PLAYER_1",
        color="GROUP",
        labels={
            "CATEGORY": "Category",
            "MARGIN_CORRELATION_PLAYER_1": "Effect on LeBron − Jordan score difference",
            "GROUP": "Section",
        },
        title="Positive generally favors LeBron; negative generally favors Jordan",
    )
    driver_figure.add_hline(y=0)
    st.plotly_chart(driver_figure, use_container_width=True)
