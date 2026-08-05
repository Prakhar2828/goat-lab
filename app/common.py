from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

RELEASE_DATA_DIR = Path("release/dashboard_data")
LOCAL_DATA_DIR = Path("data/processed")
RELEASE_MANIFEST_PATH = Path("release/v1_release_manifest.json")

CATEGORY_LABELS = {
    "peak": "Peak",
    "prime": "Prime",
    "longevity": "Longevity",
    "regular_season": "Regular season",
    "playoffs": "Playoffs",
    "offense": "Offense",
    "defense": "Defense",
    "winning_context": "Winning context",
    "cultural_impact": "Cultural impact",
}


def resolve_data_path(name: str) -> Path | None:
    """Prefer immutable release data and fall back to local build outputs."""
    for directory in (RELEASE_DATA_DIR, LOCAL_DATA_DIR):
        path = directory / name
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = resolve_data_path(name)
    if path is None:
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_release_manifest() -> dict[str, Any]:
    if not RELEASE_MANIFEST_PATH.exists():
        return {}
    payload = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def plain_language_intro(what: str, why: str, takeaway: str | None = None) -> None:
    st.markdown(f"**What this page shows:** {what}")
    st.markdown(f"**Why it matters:** {why}")
    if takeaway:
        st.info(f"**Quick takeaway:** {takeaway}")


def require_data(frame: pd.DataFrame, command: str = "make verify-release") -> None:
    if frame.empty:
        st.error(
            "Required publication data is missing. "
            f"Run `{command}` from the repository root."
        )
        st.stop()


def confidence_badge(label: str, reason: str) -> None:
    st.info(f"**Evidence confidence: {label}.** {reason}")


def friendly_category(name: str) -> str:
    return CATEGORY_LABELS.get(name, name.replace("_", " ").title())


def metric_explainer() -> None:
    with st.expander("Plain-language glossary"):
        st.markdown(
            """
- **Score difference:** the higher GOAT score minus the lower GOAT score.
- **Stress test:** many alternative scoring setups used to see whether the winner is stable.
- **Monte Carlo simulation:** the technical name for repeatedly testing randomly sampled setups.
- **Era-adjusted:** compared with the league in the same season rather than compared only by raw totals.
- **Reliability shrinkage:** small samples are pulled closer to league average so they do not look more certain than they are.
- **Expected series win probability:** how likely a team appeared to win before a playoff series began.
- **Overperformance:** actual series result minus the model's pre-series expectation.
"""
        )
