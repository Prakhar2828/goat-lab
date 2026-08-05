from pathlib import Path

import pandas as pd


def test_argument_explorer_has_no_pending_verdicts() -> None:
    arguments = pd.read_csv("data/manual/arguments.csv")
    assert not arguments["verdict"].str.strip().str.lower().eq("pending").any()
    assert arguments["relevant_page"].notna().all()


def test_dashboard_guide_exists() -> None:
    guide = Path("docs/DASHBOARD_GUIDE.md")
    assert guide.exists()
    assert "Weight simulator" in guide.read_text(encoding="utf-8")


def test_home_uses_plain_language_labels() -> None:
    text = Path("app/Home.py").read_text(encoding="utf-8")
    assert "Score difference" in text
    assert "stress test" in text.lower()
    assert "Frozen v1 leader" not in text
