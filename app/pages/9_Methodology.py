from __future__ import annotations

from pathlib import Path

import streamlit as st

from common import page_header

page_header("Methodology and limitations", "Every transformation, comparison window, assumption, and known weakness.")
for document in [
    "docs/METHODOLOGY.md",
    "docs/DATASETS.md",
    "docs/MODEL_CARD.md",
]:
    path = Path(document)
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
        st.divider()
