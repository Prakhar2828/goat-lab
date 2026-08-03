from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def season_range(start: str, end: str) -> list[str]:
    """Return NBA season strings inclusive, e.g. 1984-85 through 2025-26."""
    start_year = int(start[:4])
    end_year = int(end[:4])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def read_optional_parquet(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_parquet(source)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values_s = pd.Series(values, dtype="float64")
    weights_s = pd.Series(weights, dtype="float64")
    valid = values_s.notna() & weights_s.notna() & (weights_s > 0)
    if not valid.any():
        return float("nan")
    return float((values_s[valid] * weights_s[valid]).sum() / weights_s[valid].sum())
