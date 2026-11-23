"""Supervisor tests with a fake router and stand-in agents: no API calls, fully repeatable."""

from types import SimpleNamespace

import pytest

from dc_ai_assistant.data_assistant import agent as data
from dc_ai_assistant.rules_assistant import agent as rules
from dc_ai_assistant.supervisor.agent import FALLBACK_ANSWER, RouteDecision, _router_input, ask, build_supervisor


class FakeRouter:
    """A Responses client whose parse() returns a fixed route, or raises."""

    def __init__(self, route=None, error=None):
        self.calls = []
        self.route, self.error = route, error
        self.responses = SimpleNamespace(parse=self.parse)

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_parsed=RouteDecision(route=self.route, reason="test"))


@pytest.fixture
def agents(monkeypatch):
    """Stand-ins for the two agents that record what they were asked."""
    seen = {}

    def data_ask(question, history=None, agent=None):
        seen["data"] = (question, history)
        return {"behavior": "answer", "answer": "42 pallets", "sql": "SELECT 42", "assumptions": [], "queries_run": 1}

    def rules_ask(question, agent=None):
        seen["rules"] = question
        return {"behavior": "answer", "answer": "At least 12 mm [1]", "search_query": "q", "citations": [1]}

    monkeypatch.setattr(data, "ask", data_ask)
    monkeypatch.setattr(rules, "ask", rules_ask)
    return seen


def supervisor(router):
    return build_supervisor(client=router, data_agent=object(), rules_agent=object())


def test_routes_to_the_rules_agent(agents):
    result = ask("How tall must OVERPACK be?", supervisor=supervisor(FakeRouter("rules_agent")))
    assert result["route"] == "rules_agent"
    assert result["answer"] == "At least 12 mm [1]"
    assert agents == {"rules": "How tall must OVERPACK be?"}


def test_data_agent_gets_the_conversation(agents):
    history = [
        {"role": "user", "content": "Show me the late ones.", "route": "data_agent"},
        {"role": "assistant", "content": "Late shipments or late orders?", "route": "data_agent"},
    ]
    result = ask("Shipments.", history=history, supervisor=supervisor(FakeRouter("data_agent")))
    assert result["route"] == "data_agent"
    assert result["details"]["sql"] == "SELECT 42"
    question, sent = agents["data"]
    assert question == "Shipments." and sent == [{"role": h["role"], "content": h["content"]} for h in history]


def test_fallback_answers_without_calling_an_agent(agents):
    result = ask("Hi!", supervisor=supervisor(FakeRouter("fallback")))
    assert result["route"] == "fallback"
    assert result["behavior"] == "fallback"
    assert result["answer"] == FALLBACK_ANSWER
    assert agents == {}


def test_a_router_failure_ends_in_the_fallback(agents):
    result = ask("What was our OTIF?", supervisor=supervisor(FakeRouter(error=TimeoutError("API down"))))
    assert result["route"] == "fallback"
    assert result["reason"] == "router failed: TimeoutError"
    assert agents == {}


def test_the_router_sees_recent_turns_with_their_route():
    history = [{"role": "user", "content": f"message {i}", "route": "data_agent"} for i in range(10)]
    text = _router_input("Next one", history)
    assert "message 3" not in text and "message 4" in text  # only the last 6 turns
    assert text.endswith("New message:\nNext one")
    assert _router_input("Just this", []) == "Just this"
