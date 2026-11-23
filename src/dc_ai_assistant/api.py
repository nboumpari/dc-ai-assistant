"""HTTP API for the assistant: FastAPI, with the supervisor's steps streamed as Server-Sent Events.

Run (inside Docker, see docker-compose.yml):

    uvicorn dc_ai_assistant.api:app --host 0.0.0.0 --port 8000

Endpoints:

    GET  /api/health
    POST /api/chat/stream   {"message": "...", "history": [{"role": "user" | "assistant", "content": "...",
                                                            "route": "data_agent" | "rules_agent" | "fallback"}]}

/api/chat/stream answers with a stream of events (text/event-stream):

    event: step    {"agent": "rules_agent", "node": "retrieve", "label": "Searching the documents"}
                   sent when a step of the graph starts: the supervisor's router (agent null),
                   then the steps inside the chosen agent
    event: answer  {"route", "reason", "behavior", "answer", "details"}   the final response
    event: error   {"message": "..."}

The conversation lives in the browser: every request sends the history, so the
server keeps no session state.
"""

import json
from collections.abc import Iterator
from functools import cache
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dc_ai_assistant.supervisor.agent import build_supervisor

STEP_LABELS = {
    (None, "router"): "Choosing who should answer",
    (None, "data_agent"): "Asking the data agent",
    (None, "rules_agent"): "Asking the rules agent",
    ("data_agent", "agent"): "Working out the query",
    ("data_agent", "tools"): "Running SQL on the warehouse database",
    ("rules_agent", "reformulate"): "Rewriting the question for search",
    ("rules_agent", "retrieve"): "Searching the documents",
    ("rules_agent", "rerank"): "Ranking the passages",
    ("rules_agent", "generate"): "Writing the answer",
}


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    route: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list)


@cache
def supervisor():
    return build_supervisor()


app = FastAPI(title="DC AI Assistant", version="0.1.0")


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_chat(request: ChatRequest) -> Iterator[str]:
    state = {"question": request.message, "history": [t.model_dump() for t in request.history]}
    final: dict = {}
    try:
        for namespace, mode, chunk in supervisor().stream(state, stream_mode=["tasks", "values"], subgraphs=True):
            if mode == "values":
                if not namespace:  # the supervisor's own state, not an agent's
                    final = chunk
                continue
            if "input" not in chunk:  # a step finished; only starts are reported
                continue
            agent = namespace[0].split(":")[0] if namespace else None
            label = STEP_LABELS.get((agent, chunk["name"]))
            if label:
                yield _event("step", {"agent": agent, "node": chunk["name"], "label": label})
        yield _event(
            "answer",
            {"route": final["route"], "reason": final["reason"], **final["response"]},
        )
    except Exception as error:
        yield _event("error", {"message": f"Something went wrong ({type(error).__name__}). Please try again."})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # no buffering by proxies
    return StreamingResponse(stream_chat(request), media_type="text/event-stream", headers=headers)
