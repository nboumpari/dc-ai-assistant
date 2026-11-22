"""The data agent: a LangGraph loop that answers questions with read-only SQL.

    question -> agent (LLM) -> run_sql -> agent -> ... -> submit_answer -> done

The agent node calls OpenAI's Responses API directly (no LangChain). The
Responses API is needed because this model only allows tools together with
reasoning there. The model must call a tool on every turn: run_sql to query
the database, or submit_answer to finish. submit_answer's arguments are the
structured response (behavior, answer, sql, assumptions).

The conversation is kept as a list of Responses API input items: user and
assistant messages, the model's reasoning and function calls, and the tool
results. Reasoning items are passed back on every turn so the model keeps its
train of thought. The OpenAI client is wrapped for LangSmith, so every model
call appears in the traces. The graph is compiled on its own so the supervisor
can use it as a subgraph later.
"""

import json
import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from dc_ai_assistant import settings
from dc_ai_assistant.data_assistant.semantic_layer import render_semantic_layer
from dc_ai_assistant.data_assistant.sql_tool import SQLTool

PROMPT_PATH = Path(__file__).with_name("prompts") / "system.md"
MAX_QUERIES = 6

Item = dict[str, Any]  # a Responses API input item


class DataAgentResponse(BaseModel):
    """Final answer of the data agent."""

    behavior: Literal["answer", "clarify", "refuse", "decline"] = Field(
        description="answer = answered from the data; clarify = asking the user what they mean; "
        "refuse = request to change data or see confidential data; decline = not about the operations data."
    )
    answer: str = Field(description="The message shown to the user.")
    sql: str | None = Field(default=None, description="The query the answer is based on. Null if no query was used.")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made to answer, if any.")


TOOLS = [
    {
        "type": "function",
        "name": "run_sql",
        "description": "Run one read-only SQLite SELECT query on the warehouse database and return the result.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SQLite SELECT statement."}},
            "required": ["query"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "submit_answer",
        "description": "Submit the final answer to the user. Call this exactly once, as the last step.",
        "parameters": DataAgentResponse.model_json_schema(),
        "strict": False,
    },
]


class DataAgentState(TypedDict):
    messages: Annotated[list[Item], operator.add]
    queries_run: int
    turns: int
    response: dict[str, Any] | None


def build_system_prompt(max_queries: int = MAX_QUERIES) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{max_queries}", str(max_queries)).replace("{semantic_layer}", render_semantic_layer())


def _function_calls(items: list[Item]) -> list[Item]:
    return [i for i in items if i.get("type") == "function_call"]


def build_data_agent(client=None, sql_tool: SQLTool | None = None, max_queries: int = MAX_QUERIES):
    client = client or _default_client()
    sql_tool = sql_tool or SQLTool()
    instructions = build_system_prompt(max_queries)
    max_turns = max_queries + 3
    model_params: dict[str, Any] = {
        "model": settings.OPENAI_MODEL,
        "reasoning": {"effort": settings.OPENAI_REASONING_EFFORT},
    }
    if settings.OPENAI_REASONING_EFFORT == "none":  # the API accepts a temperature only without reasoning
        model_params["temperature"] = settings.OPENAI_TEMPERATURE

    def agent(state: DataAgentState) -> dict:
        response = client.responses.create(
            instructions=instructions, input=state["messages"], tools=TOOLS, tool_choice="required", **model_params
        )
        output = [item.model_dump(exclude_none=True) for item in response.output]
        return {"messages": output, "turns": state.get("turns", 0) + 1}

    def tools(state: DataAgentState) -> dict:
        calls = _function_calls(state["messages"][-_last_turn_size(state["messages"]) :])
        queries = state.get("queries_run", 0)
        response = None
        replies: list[Item] = []
        # run queries first, so an answer submitted in the same turn comes last
        for call in sorted(calls, key=lambda c: c["name"] == "submit_answer"):
            try:
                args = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            if args is None:
                content = "The tool arguments were not valid JSON. Call the tool again."
            elif call["name"] == "run_sql":
                if queries >= max_queries:
                    content = f"Query limit of {max_queries} reached. Call submit_answer now with what you have."
                else:
                    content = sql_tool.run(args.get("query", "")).to_text()
                    queries += 1
            elif call["name"] == "submit_answer":
                try:
                    response = DataAgentResponse(**args).model_dump()
                    content = "Answer submitted."
                except ValidationError as exc:
                    content = f"Invalid answer, call submit_answer again: {exc.errors()[0]['msg']}"
            else:
                content = f"Unknown tool '{call['name']}'. Use run_sql or submit_answer."
            replies.append({"type": "function_call_output", "call_id": call["call_id"], "output": content})

        update: dict[str, Any] = {"messages": replies, "queries_run": queries}
        if response:
            update["response"] = response
            replies.append({"role": "assistant", "content": response["answer"]})  # keeps the chat history readable
        return update

    def no_tool_call(state: DataAgentState) -> dict:
        """The model replied with plain text instead of a tool call: use it as the answer."""
        texts = [
            part.get("text", "")
            for item in state["messages"]
            if item.get("type") == "message" and item.get("role") == "assistant"
            for part in item.get("content", [])
            if isinstance(part, dict)
        ]
        text = texts[-1] if texts else ""
        return {"response": DataAgentResponse(behavior="answer", answer=text).model_dump()}

    def give_up(state: DataAgentState) -> dict:
        answer = "I could not complete this question within the allowed number of steps. Please rephrase or narrow it."
        response = DataAgentResponse(behavior="answer", answer=answer).model_dump()
        return {"response": response, "messages": [{"role": "assistant", "content": answer}]}

    def after_agent(state: DataAgentState) -> str:
        last_turn = state["messages"][-_last_turn_size(state["messages"]) :]
        return "tools" if _function_calls(last_turn) else "no_tool_call"

    def after_tools(state: DataAgentState) -> str:
        if state.get("response"):
            return END
        return "give_up" if state.get("turns", 0) >= max_turns else "agent"

    graph = StateGraph(DataAgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("no_tool_call", no_tool_call)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "no_tool_call"])
    graph.add_conditional_edges("tools", after_tools, ["agent", "give_up", END])
    graph.add_edge("no_tool_call", END)
    graph.add_edge("give_up", END)
    return graph.compile(name="data_agent")


def _last_turn_size(items: list[Item]) -> int:
    """Number of items the model produced in its latest turn (everything after the last input we sent)."""
    size = 0
    for item in reversed(items):
        if item.get("type") == "function_call_output" or item.get("role") == "user":
            break
        size += 1
    return size or 1


def _default_client():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    return wrap_openai(OpenAI())


def ask(question: str, history: list[Item] | None = None, agent=None) -> dict[str, Any]:
    """Ask one question. Returns the structured response plus how many queries were run."""
    agent = agent or build_data_agent()
    state = agent.invoke(
        {
            "messages": [*(history or []), {"role": "user", "content": question}],
            "queries_run": 0,
            "turns": 0,
            "response": None,
        }
    )
    return {**state["response"], "queries_run": state["queries_run"], "messages": state["messages"]}
