"""Runtime boundaries shared by the API and evaluation harness.

The retrieval implementation intentionally stays outside this module: teams can
keep their existing ``hybrid_retrieval.py`` (or point ``RAG_RETRIEVER_MODULE``
at another implementation) without baking an import-time model dependency into
the web service or its unit tests.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RagSettings:
    retriever_module: str = field(default_factory=lambda: os.getenv("RAG_RETRIEVER_MODULE", "hybrid_retrieval"))
    chunks_root: Path = field(default_factory=lambda: Path(os.getenv("RAG_CHUNKS_ROOT", "chunked_docs")))
    use_dense: bool = field(default_factory=lambda: os.getenv("RAG_USE_DENSE", "true").lower() == "true")
    use_reranker: bool = field(default_factory=lambda: os.getenv("RAG_USE_RERANKER", "true").lower() == "true")
    batch_size: int = field(default_factory=lambda: int(os.getenv("RAG_BATCH_SIZE", "8")))
    device: str = field(default_factory=lambda: os.getenv("RAG_DEVICE", "auto"))
    reranker_device: str = field(default_factory=lambda: os.getenv("RAG_RERANKER_DEVICE", "cpu"))
    profile: str = field(default_factory=lambda: os.getenv("RAG_PROFILE", "experimental"))


def create_retriever(settings: RagSettings | None = None) -> Any:
    """Load the configured retriever lazily and construct it using its legacy API."""
    settings = settings or RagSettings()
    try:
        module = importlib.import_module(settings.retriever_module)
        retriever_class = module.HybridRetriever
        load_chunks = module.load_chunks
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"无法加载检索核心模块 {settings.retriever_module!r}；它必须导出 HybridRetriever 和 load_chunks。"
        ) from exc

    # Older implementations do not accept a root path; retain compatibility.
    try:
        chunks = load_chunks(settings.chunks_root, include_table_children=True)
    except TypeError:
        chunks = load_chunks(include_table_children=True)
    if not chunks:
        raise RuntimeError(f"未在 {settings.chunks_root}/ 中加载到可检索文档")
    return retriever_class(
        chunks=chunks,
        use_dense=settings.use_dense,
        use_reranker=settings.use_reranker,
        batch_size=settings.batch_size,
        device=settings.device,
        reranker_device=settings.reranker_device,
        profile=settings.profile,
    )
