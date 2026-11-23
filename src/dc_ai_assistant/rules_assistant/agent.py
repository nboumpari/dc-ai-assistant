"""The rules agent: advanced RAG over official rules documents, as a LangGraph graph.

    question -> reformulate --(out of scope)--> decline
                    |
                    +--(in scope)--> retrieve -> rerank -> generate

reformulate  LLM: is the question about rules of the kind the documents cover,
             and a standalone search query for it in the documents' terms
retrieve     the reformulated query is embedded and searched (RETRIEVAL config)
rerank       the candidates are reranked against the original question, top TOP_K kept
generate     LLM: answer only from the passages, with numbered citations, or not_found

Both LLM steps call OpenAI's Responses API directly (no LangChain) with a
structured output. The OpenAI client is wrapped for LangSmith, so every model
call appears in the traces. The graph is compiled on its own so the
supervisor can use it as a subgraph later.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dc_ai_assistant import settings
from dc_ai_assistant.rules_assistant import retrieval
from dc_ai_assistant.rules_assistant.retrieval import TOP_K, RetrievalConfig

PROMPT_DIR = Path(__file__).with_name("prompts")
REFORMULATE_PROMPT = PROMPT_DIR / "reformulate.md"
ANSWER_PROMPT = PROMPT_DIR / "answer.md"
TEXT_DIR = retrieval.ROOT / "data" / "documents" / "text"

# The retrieval benchmark's winner on the dev split (evals/rules/retrieval_results.md)
RETRIEVAL = RetrievalConfig(chunking="recursive", search="vector", embedder="gemma", reranker="gte")

DECLINE_ANSWER = (
    "I can only answer questions about the rules documents: GS1 labelling standards, road transport "
    "and dangerous goods rules, EU product safety and warehouse health and safety guidance."
)


class Reformulation(BaseModel):
    in_scope: bool = Field(description="True if the question is about rules of the kind the documents cover.")
    search_query: str = Field(description="The question rewritten as a standalone search query.")


class GeneratedAnswer(BaseModel):
    behavior: Literal["answer", "not_found"] = Field(
        description="answer = answered from the passages; not_found = the passages do not contain the answer."
    )
    answer: str = Field(description="The message shown to the planner, with citations like [1].")
    citations: list[int] = Field(default_factory=list, description="Numbers of the passages the answer uses.")


class RulesAgentState(TypedDict, total=False):
    question: str
    in_scope: bool
    search_query: str
    candidates: list[int]
    passages: list[dict]
    response: dict[str, Any]


@cache
def document_titles() -> list[str]:
    return [json.loads(p.read_text(encoding="utf-8"))["title"] for p in sorted(TEXT_DIR.glob("*.json"))]


def build_reformulate_prompt() -> str:
    documents = "\n".join(f"- {title}" for title in document_titles())
    return REFORMULATE_PROMPT.read_text(encoding="utf-8").replace("{documents}", documents)


def format_passages(passages: list[dict]) -> str:
    blocks = []
    for n, chunk in enumerate(passages, start=1):
        pages = chunk.get("pages") or []
        where = "" if not pages else f", page {pages[0]}" if len(pages) == 1 else f", pages {pages[0]}-{pages[-1]}"
        blocks.append(f"[{n}] {chunk['context']}{where}\n{chunk['text']}")
    return "\n\n".join(blocks)


def model_params() -> dict[str, Any]:
    params: dict[str, Any] = {"model": settings.OPENAI_MODEL, "reasoning": {"effort": settings.OPENAI_REASONING_EFFORT}}
    if settings.OPENAI_REASONING_EFFORT == "none":  # the API accepts a temperature only without reasoning
        params["temperature"] = settings.OPENAI_TEMPERATURE
    return params


def reformulate_question(question: str, client=None) -> Reformulation:
    """The reformulate step on its own (also used by the retrieval benchmark)."""
    client = client or _default_client()
    response = client.responses.parse(
        instructions=build_reformulate_prompt(), input=question, text_format=Reformulation, **model_params()
    )
    return response.output_parsed


def build_rules_agent(client=None, config: RetrievalConfig = RETRIEVAL):
    client = client or _default_client()
    answer_instructions = ANSWER_PROMPT.read_text(encoding="utf-8")

    def reformulate(state: RulesAgentState) -> dict:
        result = reformulate_question(state["question"], client)
        return {"in_scope": result.in_scope, "search_query": result.search_query}

    def decline(state: RulesAgentState) -> dict:
        return {"passages": [], "response": {"behavior": "decline", "answer": DECLINE_ANSWER, "citations": []}}

    def retrieve(state: RulesAgentState) -> dict:
        search = RetrievalConfig(config.chunking, config.search, config.embedder)  # first stage only
        return {"candidates": retrieval.first_stage(search, state["search_query"])}

    def rerank(state: RulesAgentState) -> dict:
        ranked = state["candidates"]
        if config.reranker:
            ranked = retrieval.rerank(config.reranker, state["question"], config.chunking, ranked)
        chunks = retrieval.load_chunks(config.chunking)
        return {"passages": [chunks[i] for i in ranked[:TOP_K]]}

    def generate(state: RulesAgentState) -> dict:
        prompt = f"Question: {state['question']}\n\nPassages:\n\n{format_passages(state['passages'])}"
        response = client.responses.parse(
            instructions=answer_instructions, input=prompt, text_format=GeneratedAnswer, **model_params()
        )
        result: GeneratedAnswer = response.output_parsed
        citations = sorted({n for n in result.citations if 1 <= n <= len(state["passages"])})
        return {"response": {**result.model_dump(), "citations": citations}}

    def after_reformulate(state: RulesAgentState) -> str:
        return "retrieve" if state["in_scope"] else "decline"

    graph = StateGraph(RulesAgentState)
    graph.add_node("reformulate", reformulate)
    graph.add_node("decline", decline)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_edge(START, "reformulate")
    graph.add_conditional_edges("reformulate", after_reformulate, ["retrieve", "decline"])
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("decline", END)
    graph.add_edge("generate", END)
    return graph.compile(name="rules_agent")


def _default_client():
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    return wrap_openai(OpenAI())


def _brief(chunk: dict) -> dict:
    keys = ("chunk_id", "document", "section_id", "section_title", "pages", "char_start", "char_end")
    return {k: chunk[k] for k in keys}


def ask(question: str, agent=None) -> dict[str, Any]:
    """Ask one question. Returns the response plus the search query, the passages and the cited chunks."""
    agent = agent or build_rules_agent()
    state = agent.invoke({"question": question})
    passages = state.get("passages", [])
    response = state["response"]
    return {
        **response,
        "search_query": state.get("search_query"),
        "passages": [_brief(c) for c in passages],
        "cited_chunks": [_brief(passages[n - 1]) for n in response["citations"]],
    }
