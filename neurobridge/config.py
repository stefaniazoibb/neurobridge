"""Central configuration. Values come from environment variables or a `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROBRIDGE_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    llm_provider: Literal["none", "anthropic"] = "none"
    llm_model: str = "claude-opus-5-5"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    runs_dir: Path = REPO_ROOT / "data" / "runs"
    demo_dir: Path = REPO_ROOT / "data" / "demo"
    seed: int = 20261109
    allow_network: bool = True
    request_timeout_s: float = 120.0
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
