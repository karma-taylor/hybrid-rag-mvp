from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from harness import Identity, JwtIdentityAdapter, RagHarness, REFUSAL_ANSWER

from .auth import AuthError, AuthStore, User
from .config import Settings
from .retrieval import HybridRetriever, ROLE_LABELS

logging.basicConfig(level="INFO")
settings = Settings()
auth_store = AuthStore(settings.auth_database_path, settings.admin_initial_password)
visitors: dict[str, deque[float]] = defaultdict(deque)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=800)
    top_k: int = Field(default=5, ge=1, le=10)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=256)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class AccessAssignment(BaseModel):
    role: str
    enabled: bool = True


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Build the local retriever, LLM client and one shared security Harness once."""
    algorithms = tuple(item.strip() for item in os.getenv("RAG_JWT_ALGORITHMS", "RS256").split(",") if item.strip())
    adapter = JwtIdentityAdapter(
        environment=settings.rag_environment,
        allow_insecure_dev_auth=os.getenv("RAG_ALLOW_INSECURE_DEV_AUTH", "false").lower() == "true",
        issuer=os.getenv("RAG_JWT_ISSUER") or None,
        audience=os.getenv("RAG_JWT_AUDIENCE") or None,
        jwk_url=os.getenv("RAG_JWT_JWK_URL") or None,
        public_key=os.getenv("RAG_JWT_PUBLIC_KEY") or None,
        algorithms=algorithms,
        development_role=os.getenv("RAG_DEVELOPMENT_ROLE", "engineering"),
        jwk_timeout_seconds=float(os.getenv("RAG_JWK_TIMEOUT_SECONDS", "3")),
    )
    await adapter.prewarm()
    retriever = HybridRetriever.from_json(settings.data_path, enable_dense=settings.enable_dense, embedding_model=settings.embedding_model)
    application.state.retriever = retriever
    application.state.harness = RagHarness(
        retriever=retriever,
        llm_client=AsyncOpenAI(api_key=settings.llm_api_key or "local-missing-key", base_url=settings.llm_base_url),
        model=settings.llm_model,
        audit_salt=os.getenv("RAG_AUDIT_SALT", "local-demo-audit-salt") if settings.rag_environment == "development" else os.getenv("RAG_AUDIT_SALT", ""),
        identity_adapter=adapter,
        token_budget=settings.rag_token_budget,
    )
    yield


app = FastAPI(title="Enterprise Policy RAG Demo", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    supplied = request.headers.get("X-Request-ID", "")
    request.state.request_id = supplied if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", supplied) else str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def api_error(error: AuthError) -> None:
    raise HTTPException(error.status_code, error.args[0])


def current_user(session_id: str | None = Cookie(default=None)) -> User:
    user = auth_store.user_for_session(session_id)
    if not user:
        raise HTTPException(401, "请先登录。")
    return user


def active_member(session_id: str | None = Cookie(default=None)) -> User:
    user = current_user(session_id)
    if user.must_change_password:
        raise HTTPException(403, "首次登录请先修改密码。")
    if user.status != "active" or not user.role:
        raise HTTPException(403, "账号尚未获批访问知识库。")
    return user


def administrator(session_id: str | None = Cookie(default=None)) -> User:
    user = current_user(session_id)
    if not user.is_admin:
        raise HTTPException(403, "仅管理员可访问。")
    if user.must_change_password:
        raise HTTPException(403, "管理员首次登录请先修改密码。")
    return user


@app.get("/health")
def health(request: Request):
    harness = getattr(request.app.state, "harness", None)
    if not harness or not harness.ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready"})
    return {"status": "ok", "mode": "local-account-demo", "dense_enabled": request.app.state.retriever.enable_dense}


@app.post("/api/auth/register", status_code=201)
def register(body: Credentials):
    try:
        user = auth_store.register(body.username, body.password)
    except AuthError as error:
        api_error(error)
    return {"user": user.public(), "message": "申请已提交。请使用初始密码登录并完成首次改密，随后等待管理员分配权限。"}


@app.post("/api/auth/login")
def login(body: Credentials, response: Response):
    try:
        user = auth_store.authenticate(body.username, body.password)
    except AuthError as error:
        api_error(error)
    token = auth_store.create_session(user.id)
    response.set_cookie("session_id", token, httponly=True, samesite="strict", max_age=8 * 60 * 60, path="/")
    return {"user": user.public()}


@app.post("/api/auth/logout")
def logout(response: Response, session_id: str | None = Cookie(default=None)):
    auth_store.revoke_session(session_id)
    response.delete_cookie("session_id", path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(session_id: str | None = Cookie(default=None)):
    user = auth_store.user_for_session(session_id)
    return {"authenticated": bool(user), "user": user.public() if user else None}


@app.post("/api/auth/change-password")
def change_password(body: PasswordChange, session_id: str | None = Cookie(default=None)):
    user = current_user(session_id)
    try:
        changed = auth_store.change_password(user, body.current_password, body.new_password)
    except AuthError as error:
        api_error(error)
    return {"user": changed.public(), "message": "密码已更新。"}


@app.post("/api/v1/chat")
async def chat(body: ChatRequest, request: Request, session_id: str | None = Cookie(default=None)):
    user = active_member(session_id)
    key, now = f"{user.id}:{request.client.host if request.client else 'unknown'}", time.time()
    window = visitors[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "演示环境请求过于频繁，请稍后再试。")
    window.append(now)
    harness: RagHarness | None = getattr(request.app.state, "harness", None)
    if not harness:
        return {"answer": REFUSAL_ANSWER, "evidences": [], "retrieval": {}}
    identity = Identity(subject=f"local:{user.id}", roles=(user.role,))
    result = await harness.answer(request.state.request_id, None, body.query.strip(), body.top_k, identity=identity)
    return {"answer": result.answer, "evidences": [e.model_dump() for e in result.evidences], "retrieval": result.retrieval}


@app.get("/api/admin/users")
def users(session_id: str | None = Cookie(default=None)):
    administrator(session_id)
    return auth_store.list_users()


@app.put("/api/admin/users/{user_id}/access")
def assign_access(user_id: int, body: AccessAssignment, session_id: str | None = Cookie(default=None)):
    admin = administrator(session_id)
    if body.role not in ROLE_LABELS:
        raise HTTPException(422, "无效的业务角色。")
    try:
        user = auth_store.assign_access(admin, user_id, body.role, body.enabled)
    except AuthError as error:
        api_error(error)
    return {"user": user.public()}


@app.get("/api/admin/audit")
def audit(session_id: str | None = Cookie(default=None)):
    administrator(session_id)
    return auth_store.audit_events()


@app.get("/api/admin/documents")
def documents(request: Request, session_id: str | None = Cookie(default=None)):
    administrator(session_id)
    return [{key: doc[key] for key in ("doc_id", "title", "department", "allowed_roles")} for doc in request.app.state.retriever.documents]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/admin", include_in_schema=False)
def admin():
    return FileResponse(Path(__file__).parent / "static/admin.html")
