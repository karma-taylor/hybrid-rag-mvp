from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_path: str = os.getenv("DEMO_DATA_PATH", "data/policies.json")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    enable_dense: bool = os.getenv("RAG_ENABLE_DENSE", "false").lower() == "true"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    llm_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    llm_model: str = os.getenv("OPENAI_MODEL", "deepseek-chat")
    demo_read_only: bool = os.getenv("DEMO_READ_ONLY", "true").lower() == "true"
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "12"))
