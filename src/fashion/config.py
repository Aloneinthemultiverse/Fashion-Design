"""Configuration.

Defaults are chosen so a fresh clone runs the full test suite with no API key, no Docker
and no network: fake vision, fake embedder, in-memory store, null generation and try-on.
Real providers are opt-in through the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FASHION_", env_file=".env", extra="ignore", frozen=True
    )

    vision_provider: Literal["fake", "gemini", "qwen_local"] = "fake"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    embed_provider: Literal["fake", "openclip"] = "fake"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    store_provider: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "outfits"

    generation_provider: Literal["null", "hosted", "colab"] = "hosted"
    tryon_provider: Literal["null", "colab"] = "null"
    colab_worker_url: str = ""

    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Field(default=Path("data/seed"))
    cache_dir: Path = Field(default=Path("data/cache"))
    log_level: str = "INFO"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "labels.jsonl"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"


def load_settings() -> Settings:
    return Settings()
