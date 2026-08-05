from __future__ import annotations

from pathlib import Path

import streamlit as st
from common import page_header

page_header(
    "Methodology, results, and limitations",
    "The production freeze, transformations, assumptions, evidence rules, and known weaknesses.",
)

documents = [
    "docs/V1_FINAL_RESULTS.md",
    "docs/V1_FINAL_PREREGISTRATION.md",
    "docs/METHODOLOGY.md",
    "docs/MODEL_CARD.md",
    "docs/DATASETS.md",
]

for document in documents:
    path = Path(document)
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
        st.divider()
