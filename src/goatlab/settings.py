from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOATLAB_", env_file=".env", extra="ignore")

    start_season: str = "1984-85"
    end_season: str = "2025-26"
    request_delay_seconds: float = 4.0
    nba_timeout_seconds: int = 120
    random_seed: int = 23
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def model_dir(self) -> Path:
        return Path("models")

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.manual_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
