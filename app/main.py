from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .generation import Generator
from .retrieval import HybridRetriever, ROLE_LABELS

settings = Settings()
retriever = HybridRetriever.from_json(settings.data_path, enable_dense=settings.enable_dense, embedding_model=settings.embedding_model)
generator = Generator(settings.llm_api_key, settings.llm_base_url, settings.llm_model)
visitors: dict[str, deque[float]] = defaultdict(deque)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=800)
    user_role: Literal["guest", "engineering", "finance", "insurance", "executive"]
    top_k: int = Field(default=5, ge=1, le=10)

app = FastAPI(title="Enterprise Policy RAG Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/health")
def health(): return {"status": "ok", "mode": "public-demo", "dense_enabled": retriever.enable_dense}

@app.get("/api/roles")
def roles(): return ROLE_LABELS

@app.post("/api/v1/chat")
def chat(body: ChatRequest, request: Request):
    key = request.client.host if request.client else "unknown"
    now = time.time(); window = visitors[key]
    while window and now - window[0] > 60: window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "演示环境请求过于频繁，请稍后再试")
    window.append(now)
    evidences = retriever.search(body.query.strip(), body.user_role, body.top_k)
    answer, mode = generator.answer(body.query, evidences)
    return {"query": body.query, "answer": answer, "evidences": [e.__dict__ for e in evidences], "generation_mode": mode}

@app.get("/api/admin/documents")
def documents():
    return [{k: d[k] for k in ("doc_id", "title", "department", "allowed_roles")} for d in retriever.documents]

@app.post("/api/admin/documents")
def upload_demo_guard():
    if settings.demo_read_only: raise HTTPException(403, "公开演示环境为只读模式；本地可关闭 DEMO_READ_ONLY 后接入上传管道")
    raise HTTPException(501, "作品版仅演示治理流程，生产环境应接入审核后的解析与索引任务")

@app.get("/", include_in_schema=False)
def index(): return FileResponse(Path(__file__).parent / "static/index.html")
@app.get("/xray", include_in_schema=False)
def xray(): return FileResponse(Path(__file__).parent / "static/xray.html")
@app.get("/admin", include_in_schema=False)
def admin(): return FileResponse(Path(__file__).parent / "static/admin.html")
