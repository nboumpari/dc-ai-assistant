"""The data agent: a LangGraph loop that answers questions with read-only SQL.

    question -> agent (LLM) -> run_sql -> agent -> ... -> submit_answer -> done

The agent node calls the OpenAI API directly (no LangChain). The model must
call a tool on every turn: run_sql to query the database, or submit_answer to
finish. submit_answer's arguments are the structured response (behavior,
answer, sql, assumptions), so no extra call is needed to format the result.
Messages are plain OpenAI-format dicts. The OpenAI client is wrapped for
LangSmith, so every model call appears in the traces. The graph is compiled on
its own so the supervisor can use it as a subgraph later.
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

Message = dict[str, Any]  # OpenAI chat message


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
        "function": {
            "name": "run_sql",
            "description": "Run one read-only SQLite SELECT query on the warehouse database and return the result.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A single SQLite SELECT statement."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer to the user. Call this exactly once, as the last step.",
            "parameters": DataAgentResponse.model_json_schema(),
        },
    },
]


class DataAgentState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    queries_run: int
    turns: int
    response: dict[str, Any] | None


def build_system_prompt(max_queries: int = MAX_QUERIES) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{max_queries}", str(max_queries)).replace("{semantic_layer}", render_semantic_layer())


def _assistant_message(message) -> Message:
    """Convert the SDK's response message into a plain dict we can send back."""
    out: Message = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return out


def build_data_agent(client=None, sql_tool: SQLTool | None = None, max_queries: int = MAX_QUERIES):
    client = client or _default_client()
    sql_tool = sql_tool or SQLTool()
    system: Message = {"role": "system", "content": build_system_prompt(max_queries)}
    max_turns = max_queries + 3
    model_params: dict[str, Any] = {"model": settings.OPENAI_MODEL, "temperature": settings.OPENAI_TEMPERATURE}
    if settings.OPENAI_REASONING_EFFORT:
        model_params["reasoning_effort"] = settings.OPENAI_REASONING_EFFORT

    def agent(state: DataAgentState) -> dict:
        completion = client.chat.completions.create(
            messages=[system, *state["messages"]], tools=TOOLS, tool_choice="required", **model_params
        )
        return {"messages": [_assistant_message(completion.choices[0].message)], "turns": state.get("turns", 0) + 1}

    def tools(state: DataAgentState) -> dict:
        calls = state["messages"][-1].get("tool_calls", [])
        queries = state.get("queries_run", 0)
        response = None
        replies: list[Message] = []
        # run queries first, so an answer submitted in the same turn comes last
        for call in sorted(calls, key=lambda c: c["function"]["name"] == "submit_answer"):
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            if args is None:
                content = "The tool arguments were not valid JSON. Call the tool again."
            elif name == "run_sql":
                if queries >= max_queries:
                    content = f"Query limit of {max_queries} reached. Call submit_answer now with what you have."
                else:
                    content = sql_tool.run(args.get("query", "")).to_text()
                    queries += 1
            elif name == "submit_answer":
                try:
                    response = DataAgentResponse(**args).model_dump()
                    content = "Answer submitted."
                except ValidationError as exc:
                    content = f"Invalid answer, call submit_answer again: {exc.errors()[0]['msg']}"
            else:
                content = f"Unknown tool '{name}'. Use run_sql or submit_answer."
            replies.append({"role": "tool", "tool_call_id": call["id"], "content": content})

        update: dict[str, Any] = {"messages": replies, "queries_run": queries}
        if response:
            update["response"] = response
            replies.append({"role": "assistant", "content": response["answer"]})  # keeps the chat history readable
        return update

    def no_tool_call(state: DataAgentState) -> dict:
        """The model replied with plain text instead of a tool call: use it as the answer."""
        text = state["messages"][-1].get("content") or ""
        return {"response": DataAgentResponse(behavior="answer", answer=text).model_dump()}

    def give_up(state: DataAgentState) -> dict:
        answer = "I could not complete this question within the allowed number of steps. Please rephrase or narrow it."
        response = DataAgentResponse(behavior="answer", answer=answer).model_dump()
        return {"response": response, "messages": [{"role": "assistant", "content": answer}]}

    def after_agent(state: DataAgentState) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") else "no_tool_call"

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


def _default_client():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    return wrap_openai(OpenAI())


def ask(question: str, history: list[Message] | None = None, agent=None) -> dict[str, Any]:
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
