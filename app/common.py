from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

RELEASE_DATA_DIR = Path("release/dashboard_data")
LOCAL_DATA_DIR = Path("data/processed")
RELEASE_MANIFEST_PATH = Path("release/v1_release_manifest.json")


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


def require_data(frame: pd.DataFrame, command: str = "make verify-release") -> None:
    if frame.empty:
        st.error(
            "Required publication data is missing. "
            f"Run `{command}` from the repository root."
        )
        st.stop()


def confidence_badge(label: str, reason: str) -> None:
    st.info(f"**Evidence confidence: {label}.** {reason}")


def metric_explainer() -> None:
    with st.expander("How to read the scores"):
        st.write(
            "Scores are evidence summaries, not universal truths. Every result should be read "
            "with its comparison window, source coverage, uncertainty, and selected value weights."
        )
