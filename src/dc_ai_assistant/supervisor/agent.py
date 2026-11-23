"""The supervisor: a LangGraph graph that checks scope and routes each message to one agent.

    message -> router --> data_agent   the data agent (text-to-SQL over the warehouse database)
                     +--> rules_agent  the rules agent (advanced RAG over official documents)
                     +--> fallback     a fixed reply: what the assistant can help with

The router is one LLM call (OpenAI Responses API, structured output) that sees
the new message and the recent conversation. It never answers a question
itself, and the answers of the two agents are never merged. If the router
fails (an API error or an unusable reply), the message goes to the fallback,
so the user always gets a reply. The two agents are the same graphs that were
evaluated on their own.
"""

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dc_ai_assistant import settings

PROMPT_PATH = Path(__file__).with_name("prompts") / "router.md"
HISTORY_TURNS = 6  # earlier messages the router sees

Route = Literal["data_agent", "rules_agent", "fallback"]

FALLBACK_ANSWER = (
    "I'm the control tower assistant. I can help with two kinds of questions:\n"
    "- Our 2025 warehouse operations, answered from the database: orders, shipments, pallets, stock, "
    'carriers, dock appointments and KPIs such as OTIF. For example: "What was our OTIF in June?"\n'
    "- Rules and standards for warehousing and transport: GS1 labels and barcodes, dangerous goods (ADR), "
    "CMR, drivers' hours, product safety and warehouse health and safety. For example: "
    '"How tall must the word OVERPACK be on a limited-quantity overpack?"\n'
    "If your question is about one of these, please rephrase it."
)


class RouteDecision(BaseModel):
    route: Route = Field(description="Who handles the message.")
    reason: str = Field(description="One short sentence on why.")


class SupervisorState(TypedDict, total=False):
    question: str
    history: list[dict[str, Any]]  # earlier turns: {"role": "user" | "assistant", "content": str, "route": ...}
    route: Route
    reason: str
    response: dict[str, Any]


def _model_params() -> dict[str, Any]:
    params: dict[str, Any] = {"model": settings.OPENAI_MODEL, "reasoning": {"effort": settings.OPENAI_REASONING_EFFORT}}
    if settings.OPENAI_REASONING_EFFORT == "none":  # the API accepts a temperature only without reasoning
        params["temperature"] = settings.OPENAI_TEMPERATURE
    return params


def _router_input(question: str, history: list[dict[str, Any]] | None) -> str:
    turns = (history or [])[-HISTORY_TURNS:]
    if not turns:
        return question
    lines = []
    for turn in turns:
        who = "User" if turn["role"] == "user" else f"Assistant ({turn.get('route', 'unknown')})"
        lines.append(f"{who}: {turn['content']}")
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nNew message:\n{question}"


def route_question(question: str, history: list[dict[str, Any]] | None = None, client=None) -> RouteDecision:
    """The router on its own (also used by the router evaluation)."""
    client = client or _default_client()
    response = client.responses.parse(
        instructions=PROMPT_PATH.read_text(encoding="utf-8"),
        input=_router_input(question, history),
        text_format=RouteDecision,
        **_model_params(),
    )
    return response.output_parsed


def build_supervisor(client=None, data_agent=None, rules_agent=None):
    """The agents are built when first needed, so a message for one agent never loads the other."""
    from dc_ai_assistant.data_assistant import agent as data
    from dc_ai_assistant.rules_assistant import agent as rules

    client = client or _default_client()
    agents: dict[str, Any] = {"data": data_agent, "rules": rules_agent}

    def router(state: SupervisorState) -> dict:
        try:
            decision = route_question(state["question"], state.get("history"), client)
            if decision is None:
                raise ValueError("no route in the reply")
        except Exception as error:  # any router failure ends in the fallback, never in a crash
            return {"route": "fallback", "reason": f"router failed: {type(error).__name__}"}
        return {"route": decision.route, "reason": decision.reason}

    def data_agent_node(state: SupervisorState) -> dict:
        agents["data"] = agents["data"] or data.build_data_agent(client=client)
        history = [{"role": t["role"], "content": t["content"]} for t in state.get("history") or []]
        result = data.ask(state["question"], history=history, agent=agents["data"])
        details = {k: result.get(k) for k in ("sql", "assumptions", "queries_run")}
        return {"response": {"behavior": result["behavior"], "answer": result["answer"], "details": details}}

    def rules_agent_node(state: SupervisorState) -> dict:
        agents["rules"] = agents["rules"] or rules.build_rules_agent(client=client)
        result = rules.ask(state["question"], agent=agents["rules"])
        details = {k: result.get(k) for k in ("search_query", "citations", "cited_chunks")}
        return {"response": {"behavior": result["behavior"], "answer": result["answer"], "details": details}}

    def fallback(state: SupervisorState) -> dict:
        return {"response": {"behavior": "fallback", "answer": FALLBACK_ANSWER, "details": {}}}

    graph = StateGraph(SupervisorState)
    graph.add_node("router", router)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("rules_agent", rules_agent_node)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda state: state["route"], ["data_agent", "rules_agent", "fallback"])
    for node in ("data_agent", "rules_agent", "fallback"):
        graph.add_edge(node, END)
    return graph.compile(name="supervisor")


def _default_client():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    return wrap_openai(OpenAI())


def ask(question: str, history: list[dict[str, Any]] | None = None, supervisor=None) -> dict[str, Any]:
    """Ask one message. Returns the route, the router's reason and the chosen agent's response."""
    supervisor = supervisor or build_supervisor()
    state = supervisor.invoke({"question": question, "history": history or []})
    return {"route": state["route"], "reason": state["reason"], **state["response"]}
