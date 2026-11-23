"""API tests: the streamed events, with a fake router and stand-in agents (no API calls)."""

import json

import pytest
from fastapi.testclient import TestClient

from dc_ai_assistant import api
from dc_ai_assistant.supervisor.agent import build_supervisor
from tests.test_supervisor import FakeRouter, agents  # noqa: F401  (agents is a fixture)


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def client(monkeypatch, agents):  # noqa: F811
    def use(router):
        graph = build_supervisor(client=router, data_agent=object(), rules_agent=object())
        monkeypatch.setattr(api, "supervisor", lambda: graph)
        return TestClient(api.app)

    return use


def test_streams_the_steps_then_the_answer(client):
    response = client(FakeRouter("rules_agent")).post("/api/chat/stream", json={"message": "How tall is OVERPACK?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    got = events(response)
    assert [e for e, _ in got] == ["step", "step", "answer"]
    assert got[0][1]["label"] == "Choosing who should answer"
    assert got[1][1]["label"] == "Asking the rules agent"
    assert got[2][1]["route"] == "rules_agent" and got[2][1]["answer"] == "At least 12 mm [1]"


def test_history_reaches_the_agent(client, agents):  # noqa: F811
    history = [{"role": "user", "content": "Show me the late ones.", "route": "data_agent"}]
    response = client(FakeRouter("data_agent")).post(
        "/api/chat/stream", json={"message": "Shipments.", "history": history}
    )
    assert events(response)[-1][1]["details"]["sql"] == "SELECT 42"
    assert agents["data"] == ("Shipments.", [{"role": "user", "content": "Show me the late ones."}])


def test_an_agent_failure_becomes_an_error_event(client, monkeypatch):
    from dc_ai_assistant.rules_assistant import agent as rules

    def broken(question, agent=None):
        raise RuntimeError("Qdrant is down")

    monkeypatch.setattr(rules, "ask", broken)
    got = events(client(FakeRouter("rules_agent")).post("/api/chat/stream", json={"message": "x"}))
    assert got[-1][0] == "error"
    assert "Qdrant is down" not in got[-1][1]["message"]  # internal details stay on the server


def test_an_empty_message_is_rejected(client):
    assert client(FakeRouter("fallback")).post("/api/chat/stream", json={"message": ""}).status_code == 422
