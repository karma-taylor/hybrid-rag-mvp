from uuid import uuid4

from fastapi.testclient import TestClient

from app.auth import User
from app.main import app, auth_store
from harness import Evidence, HarnessResult


class RecordingHarness:
    ready = True

    def __init__(self) -> None:
        self.identity = None

    async def answer(self, request_id, authorization, query, top_k, *, identity=None):
        self.identity = identity
        return HarnessResult("工程制度要求审批[1]。", [Evidence(doc_id="ENG-001", text="工程制度", score=1)], {}, None)


def prepare_member(client: TestClient, role: str = "engineering") -> TestClient:
    username = f"member-{uuid4().hex[:12]}"
    password = "StartPass!2026"
    assert client.post("/api/auth/register", json={"username": username, "password": password}).status_code == 201
    assert client.post("/api/auth/login", json={"username": username, "password": password}).status_code == 200
    assert client.post("/api/auth/change-password", json={"current_password": password, "new_password": "ChangedPass!2026"}).status_code == 200
    user = auth_store.authenticate(username, "ChangedPass!2026")
    auth_store.assign_access(User(0, "test-admin", None, "active", False, True), user.id, role, True)
    return client


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_engineering_can_retrieve_own_policy():
    with TestClient(app) as client:
        harness = RecordingHarness()
        app.state.harness = harness
        prepare_member(client)
        response = client.post("/api/v1/chat", json={"query": "代码合并前需要什么批准？", "top_k": 5})
        assert response.status_code == 200 and response.json()["evidences"][0]["doc_id"] == "ENG-001"
        assert harness.identity.roles == ("engineering",)


def test_pending_user_is_fail_closed():
    with TestClient(app) as client:
        username = f"pending-{uuid4().hex[:12]}"
        password = "PendingPass!2026"
        client.post("/api/auth/register", json={"username": username, "password": password})
        client.post("/api/auth/login", json={"username": username, "password": password})
        client.post("/api/auth/change-password", json={"current_password": password, "new_password": "ChangedPass!2026"})
        response = client.post("/api/v1/chat", json={"query": "差旅费用多久内提交报销？"})
        assert response.status_code == 403


def test_admin_endpoint_requires_administrator():
    assert TestClient(app).get("/api/admin/users").status_code == 401
