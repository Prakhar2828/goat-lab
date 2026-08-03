from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from goatlab.settings import settings
from goatlab.utils import stable_hash


NBA_STATS_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
}


def patch_nba_api_headers() -> None:
    """Apply browser-like headers and reset nba_api's shared session."""
    try:
        from nba_api.library import http as base_http
        from nba_api.stats.library import http as stats_http

        stats_http.STATS_HEADERS = NBA_STATS_HEADERS
        stats_http.NBAStatsHTTP.headers = NBA_STATS_HEADERS
        stats_http.NBAStatsHTTP._session = None
        base_http.NBAHTTP._session = None
    except Exception:
        # Passing headers directly to each endpoint remains the fallback.
        pass


class NbaApiError(RuntimeError):
    pass


class CachedNbaClient:
    """Small cache and retry layer around nba_api endpoint classes.

    NBA Stats requests can time out or reject repeated calls. Every response is cached,
    calls are delayed, and failures are retried with jitter.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        patch_nba_api_headers()
        self.cache_dir = Path(cache_dir or settings.raw_dir / "nba_api")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(6),
        wait=wait_exponential_jitter(initial=3, max=60),
        reraise=True,
    )
    def fetch(
        self,
        endpoint_factory: Callable[..., Any],
        endpoint_name: str,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        cache_key = stable_hash({"endpoint": endpoint_name, "kwargs": kwargs})
        cache_path = self.cache_dir / endpoint_name / f"{cache_key}.json"
        if cache_path.exists():
            return self._read_cache(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            endpoint = endpoint_factory(
                timeout=settings.nba_timeout_seconds,
                headers=NBA_STATS_HEADERS,
                **kwargs,
            )
            frames = endpoint.get_data_frames()
            names = self._dataset_names(endpoint, len(frames))
            payload = {
                name: frame.where(pd.notna(frame), None).to_dict(orient="records")
                for name, frame in zip(names, frames, strict=False)
            }
            cache_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
            time.sleep(settings.request_delay_seconds)
            return {name: pd.DataFrame(rows) for name, rows in payload.items()}
        except Exception as exc:  # pragma: no cover - network behavior
            raise NbaApiError(f"{endpoint_name} failed for {kwargs}: {exc}") from exc

    @staticmethod
    def _dataset_names(endpoint: Any, count: int) -> list[str]:
        names: list[str] = []
        data_sets = getattr(endpoint, "data_sets", None)
        if isinstance(data_sets, dict):
            names = list(data_sets.keys())
        if len(names) != count:
            names = [f"dataset_{index}" for index in range(count)]
        return names

    @staticmethod
    def _read_cache(path: Path) -> dict[str, pd.DataFrame]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {name: pd.DataFrame(rows) for name, rows in payload.items()}
