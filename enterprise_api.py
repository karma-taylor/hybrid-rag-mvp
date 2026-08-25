"""HTTP boundary for the security-first RAG Harness."""
from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

from harness import (
    REFUSAL_ANSWER,
    Evidence,
    JwtIdentityAdapter,
    RagHarness,
)
from harness import (
    answer_has_only_valid_citations as _citation_validator,
)
from harness import (
    build_rag_prompt as _build_rag_prompt,
)
from rag_runtime import RagSettings, create_retriever

logger = logging.getLogger("enterprise_rag_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Compatibility exports for callers that previously tested prompt validation here.
_answer_has_only_valid_citations = _citation_validator
build_rag_prompt = _build_rag_prompt


class ChatRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=2_000)]
    top_k: Annotated[int, Field(default=5, ge=1, le=10)]

    @field_validator("query")
    @classmethod
    def strip_required_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空")
        return value


class ChatResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    evidences: list[Evidence]
    retrieval: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize dependencies once; leave process live on failed startup."""
    try:
        adapter = JwtIdentityAdapter.from_environment()
        await adapter.prewarm()
        app.state.harness = RagHarness(
            retriever=create_retriever(RagSettings()),
            llm_client=AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            ),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            audit_salt=os.getenv("RAG_AUDIT_SALT", ""),
            identity_adapter=adapter,
            token_budget=int(os.getenv("RAG_EVIDENCE_TOKEN_BUDGET", "8000")),
            retrieval_timeout_seconds=float(os.getenv("RAG_RETRIEVAL_TIMEOUT_SECONDS", "10")),
            generation_timeout_seconds=float(os.getenv("RAG_GENERATION_TIMEOUT_SECONDS", "20")),
            total_timeout_seconds=float(os.getenv("RAG_TOTAL_TIMEOUT_SECONDS", "30")),
            retrieval_concurrency=int(os.getenv("RAG_RETRIEVAL_CONCURRENCY", "4")),
            generation_concurrency=int(os.getenv("RAG_GENERATION_CONCURRENCY", "4")),
            circuit_failure_threshold=int(os.getenv("RAG_CIRCUIT_FAILURE_THRESHOLD", "3")),
            circuit_cooldown_seconds=float(os.getenv("RAG_CIRCUIT_COOLDOWN_SECONDS", "30")),
        )
        logger.info("Enterprise RAG Harness initialized")
    except Exception:
        logger.exception("Enterprise RAG Harness initialization failed")
        app.state.harness = None
    yield


app = FastAPI(
    title="Enterprise Hybrid RAG API",
    version="2.0.0",
    description="ACL-first retrieval and generation governed by RagHarness",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    supplied_id = request.headers.get("X-Request-ID", "")
    request.state.request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", supplied_id) else str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health(request: Request) -> Response:
    if request.app.state.harness is None or not request.app.state.harness.ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready"})
    return JSONResponse(content={"status": "ok"})


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Delegate all authentication and RAG behavior to the Harness."""
    request_id = request.state.request_id
    harness: RagHarness | None = request.app.state.harness
    if harness is None:
        return ChatResponse(request_id=request_id, query=payload.query, answer=REFUSAL_ANSWER, evidences=[], retrieval={})
    result = await harness.answer(request_id, request.headers.get("Authorization"), payload.query, payload.top_k)
    return ChatResponse(request_id=request_id, query=payload.query, answer=result.answer, evidences=result.evidences, retrieval=result.retrieval)
