from __future__ import annotations

from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from goatlab.settings import settings
from goatlab.utils import write_parquet


WIKIMEDIA_ENDPOINT = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}"
)


def fetch_wikimedia_pageviews(
    article: str,
    start: str = "20150701",
    end: str | None = None,
) -> pd.DataFrame:
    end = end or date.today().strftime("%Y%m%d")
    url = WIKIMEDIA_ENDPOINT.format(article=quote(article, safe=""), start=start, end=end)
    response = requests.get(
        url,
        headers={"User-Agent": "GOAT-Lab/0.1 research project contact@example.com"},
        timeout=60,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["timestamp"].str[:8], format="%Y%m%d")
    frame["article_requested"] = article
    return frame[["article_requested", "article", "date", "views"]]


def ingest_wikimedia() -> pd.DataFrame:
    frames = []
    for player, article in {
        "Michael Jordan": "Michael_Jordan",
        "LeBron James": "LeBron_James",
    }.items():
        frame = fetch_wikimedia_pageviews(article)
        frame["PLAYER_NAME"] = player
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    write_parquet(result, settings.interim_dir / "wikimedia_pageviews.parquet")
    return result


def import_google_trends_csv(path: str | Path, player_name: str) -> pd.DataFrame:
    """Import a Google Trends CSV manually exported from trends.google.com.

    Manual export is preferred over relying on unofficial pytrends endpoints.
    """
    frame = pd.read_csv(path, skiprows=1)
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = frame.columns[0]
    value_columns = frame.columns[1:]
    long = frame.melt(id_vars=[date_column], value_vars=value_columns, var_name="query", value_name="interest")
    long = long.rename(columns={date_column: "date"})
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["interest"] = pd.to_numeric(long["interest"], errors="coerce")
    long["PLAYER_NAME"] = player_name
    return long.dropna(subset=["date", "interest"])
