from __future__ import annotations

from pathlib import Path

import pandas as pd

from goatlab.settings import settings
from goatlab.utils import write_parquet

EXPECTED_ADVANCED_COLUMNS = {
    "player",
    "season",
    "season_type",
    "per",
    "ts_pct",
    "usg_pct",
    "ows",
    "dws",
    "ws",
    "ws_per_48",
    "obpm",
    "dbpm",
    "bpm",
    "vorp",
}


def import_manual_advanced(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    normalized = {column.lower().strip() for column in frame.columns}
    missing = EXPECTED_ADVANCED_COLUMNS - normalized
    if missing:
        raise ValueError(f"Manual advanced CSV is missing columns: {sorted(missing)}")
    frame.columns = [column.lower().strip() for column in frame.columns]
    write_parquet(frame, settings.interim_dir / "manual_advanced.parquet")
    return frame


def import_mvp_votes(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"player", "season", "rank", "first_place_votes", "points_won", "points_max", "share"}
    frame.columns = [column.lower().strip() for column in frame.columns]
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MVP voting CSV is missing columns: {sorted(missing)}")
    write_parquet(frame, settings.interim_dir / "mvp_votes.parquet")
    return frame
