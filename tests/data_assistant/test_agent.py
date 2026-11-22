"""Agent loop tests with a scripted fake OpenAI Responses client: no API calls, fully repeatable."""

import json
from types import SimpleNamespace

import pytest

from dc_ai_assistant.data_assistant.agent import ask, build_data_agent, build_system_prompt
from dc_ai_assistant.data_assistant.sql_tool import SQLTool


class Item(SimpleNamespace):
    """An output item of the Responses API."""

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def tool_call(name, args, call_id):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return Item(type="function_call", call_id=call_id, name=name, arguments=arguments)


def reply(*items, text=None, reasoning=True):
    output = [Item(type="reasoning", id="rs_1", summary=[])] if reasoning else []
    output += list(items)
    if text is not None:
        output.append(Item(type="message", role="assistant", content=[{"type": "output_text", "text": text}]))
    return output


def sql(query, call_id="q1"):
    return reply(tool_call("run_sql", {"query": query}, call_id))


def submit(behavior, answer, sql_text=None, call_id="s1"):
    args = {"behavior": behavior, "answer": answer, "sql": sql_text, "assumptions": []}
    return reply(tool_call("submit_answer", args, call_id))


class FakeOpenAI:
    """Mimics client.responses.create: returns scripted output items and records each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        output = self.replies.pop(0) if self.replies else submit("answer", "out of script")
        return SimpleNamespace(output=output)

    def last_tool_reply(self, request=-1):
        outputs = [i for i in self.requests[request]["input"] if i.get("type") == "function_call_output"]
        return outputs[-1]["output"]


@pytest.fixture
def sql_tool(small_db_files):
    return SQLTool(db_path=small_db_files[0])


def run(replies, question="Q", sql_tool=None, **kwargs):
    client = FakeOpenAI(replies)
    result = ask(question, agent=build_data_agent(client=client, sql_tool=sql_tool, **kwargs))
    return client, result


def test_query_then_answer(sql_tool):
    client, result = run(
        [sql("SELECT COUNT(*) FROM warehouses"), submit("answer", "4 sites.", "SELECT COUNT(*) FROM warehouses")],
        sql_tool=sql_tool,
    )
    assert result["behavior"] == "answer"
    assert result["sql"] == "SELECT COUNT(*) FROM warehouses"
    assert result["queries_run"] == 1
    assert "(1 rows)" in client.last_tool_reply()


def test_request_settings(sql_tool):
    client, _ = run([submit("clarify", "Which rate?")], sql_tool=sql_tool)
    request = client.requests[0]
    assert request["tool_choice"] == "required"
    assert [t["name"] for t in request["tools"]] == ["run_sql", "submit_answer"]
    assert request["reasoning"] == {"effort": "low"}
    assert "temperature" not in request  # not allowed together with reasoning
    assert "### pallets" in request["instructions"]


def test_clarify_without_queries(sql_tool):
    _, result = run([submit("clarify", "Late deliveries, transfers or dock arrivals?")], sql_tool=sql_tool)
    assert result["behavior"] == "clarify"
    assert result["queries_run"] == 0


def test_blocked_query_error_reaches_the_model(sql_tool):
    client, result = run(
        [sql("SELECT cost_per_pallet FROM carriers"), submit("refuse", "Confidential.")], sql_tool=sql_tool
    )
    assert "Not allowed" in client.last_tool_reply()
    assert result["behavior"] == "refuse"


def test_query_limit_is_enforced(sql_tool):
    replies = [sql("SELECT 1", f"q{i}") for i in range(4)] + [submit("answer", "done")]
    client, result = run(replies, sql_tool=sql_tool, max_queries=2)
    assert result["queries_run"] == 2
    assert "Query limit of 2 reached" in client.last_tool_reply()


def test_gives_up_after_too_many_turns(sql_tool):
    _, result = run([sql("SELECT 1", f"q{i}") for i in range(20)], sql_tool=sql_tool, max_queries=2)
    assert "could not complete" in result["answer"]


def test_invalid_submission_is_sent_back(sql_tool):
    bad = reply(tool_call("submit_answer", {"behavior": "maybe"}, "b1"))
    client, result = run([bad, submit("answer", "fixed")], sql_tool=sql_tool)
    assert "Invalid answer" in client.last_tool_reply()
    assert result["answer"] == "fixed"


def test_broken_json_arguments_are_sent_back(sql_tool):
    broken = reply(tool_call("run_sql", "{not json", "j1"))
    client, result = run([broken, submit("answer", "ok")], sql_tool=sql_tool)
    assert "not valid JSON" in client.last_tool_reply()
    assert result["queries_run"] == 0


def test_plain_text_reply_becomes_the_answer(sql_tool):
    _, result = run([reply(text="Just text")], sql_tool=sql_tool)
    assert result["answer"] == "Just text"


def test_follow_up_sees_the_history(sql_tool):
    client = FakeOpenAI([submit("answer", "first"), submit("answer", "second")])
    agent = build_data_agent(client=client, sql_tool=sql_tool)
    first = ask("Q1", agent=agent)
    ask("And for Patras?", history=first["messages"], agent=agent)
    sent = [m.get("content") for m in client.requests[1]["input"]]
    assert "Q1" in sent and "first" in sent and "And for Patras?" in sent


def test_system_prompt_is_filled_in():
    prompt = build_system_prompt(max_queries=6)
    assert "{" + "semantic_layer}" not in prompt and "{" + "max_queries}" not in prompt
    assert "at most 6 queries" in prompt
    assert "### pallets" in prompt


def test_reasoning_items_are_sent_back(sql_tool):
    client, _ = run([sql("SELECT 1"), submit("answer", "ok")], sql_tool=sql_tool)
    second_input = client.requests[1]["input"]
    types = [i.get("type") for i in second_input]
    assert types[-3:] == ["reasoning", "function_call", "function_call_output"]
