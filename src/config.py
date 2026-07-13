from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    vector_store_path: str = "vector_db"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    temperature: float = 0.2

    @property
    def api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")


def load_config(path: str | Path) -> RAGConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return RAGConfig(**data)