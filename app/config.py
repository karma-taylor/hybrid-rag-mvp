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
    auth_database_path: str = os.getenv("AUTH_DATABASE_PATH", "data/rag_demo.sqlite")
    # The local-demo bootstrap password is deliberately forced to rotate at first sign-in.
    admin_initial_password: str = os.getenv("ADMIN_INITIAL_PASSWORD", "Admin@123456")
    rag_environment: str = os.getenv("RAG_ENV", "development").lower()
    rag_audit_salt: str = os.getenv("RAG_AUDIT_SALT", "local-demo-audit-salt")
    rag_token_budget: int = int(os.getenv("RAG_EVIDENCE_TOKEN_BUDGET", "2000"))
