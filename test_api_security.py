
from fastapi.testclient import TestClient

from enterprise_api import app
from harness import REFUSAL_ANSWER, HarnessResult


class RefusingHarness:
    ready = True

    async def answer(self, request_id, authorization, query, top_k):
        return HarnessResult(REFUSAL_ANSWER, [], {}, "AUTH_INVALID")


def test_invalid_client_request_id_is_replaced() -> None:
    app.state.harness = RefusingHarness()
    client = TestClient(app)
    response = client.post("/api/v1/chat", headers={"X-Request-ID": "bad id with spaces"}, json={"query": "查询"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "bad id with spaces"
    assert response.json()["answer"] == REFUSAL_ANSWER
