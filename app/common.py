from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data/processed")


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def require_data(frame: pd.DataFrame, command: str = "make all") -> None:
    if frame.empty:
        st.error(f"Required processed data is missing. Run `{command}` from the repository root.")
        st.stop()


def confidence_badge(label: str, reason: str) -> None:
    st.info(f"**Evidence confidence: {label}.** {reason}")


def metric_explainer() -> None:
    with st.expander("How to read the scores"):
        st.write(
            "Scores are evidence summaries, not universal truths. Every result should be read "
            "with its comparison window, source coverage, uncertainty, and selected value weights."
        )
