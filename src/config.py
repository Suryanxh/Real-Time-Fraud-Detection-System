"""Application configuration and environment-aware defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Centralized settings for the demo and production-friendly deployment."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    api_base_url: str = "http://localhost:8000"
    api_key: str | None = None
    event_delay: float = 0.5
    replay_limit: int = 50

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def model_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def score_endpoint(self) -> str:
        return f"{self.api_base_url}/score"

    @property
    def health_endpoint(self) -> str:
        return f"{self.api_base_url}/health"

    @property
    def requires_api_key(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        api_base_url = os.getenv("FRAUD_API_BASE_URL", "http://localhost:8000")
        api_key = os.getenv("FRAUD_API_KEY") or None
        event_delay = float(os.getenv("STREAM_EVENT_DELAY", "0.5"))
        replay_limit = int(os.getenv("STREAM_LIMIT_ROWS", "50"))
        return cls(
            api_base_url=api_base_url,
            api_key=api_key,
            event_delay=event_delay,
            replay_limit=replay_limit,
        )


settings = Settings.from_env()
