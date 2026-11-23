"""Minimal Qdrant client over its REST API.

One collection per chunking strategy (rules_fixed, rules_recursive,
rules_structure). Each point is one chunk with one named vector per embedding
model ("gemma", "granite") and the chunk itself as payload.
"""

import requests

from dc_ai_assistant import settings

TIMEOUT = 60


def collection_name(strategy: str) -> str:
    return f"rules_{strategy}"


def _call(method: str, path: str, body: dict | None = None) -> dict:
    response = requests.request(method, f"{settings.QDRANT_URL}{path}", json=body, timeout=TIMEOUT)
    if not response.ok:
        raise RuntimeError(f"Qdrant {method} {path} failed ({response.status_code}): {response.text[:300]}")
    return response.json()


def recreate_collection(name: str, vectors: dict[str, int]) -> None:
    """Drop the collection if it exists, then create it with cosine named vectors {name: dims}."""
    if name in list_collections():
        _call("DELETE", f"/collections/{name}")
    config = {vector: {"size": dims, "distance": "Cosine"} for vector, dims in vectors.items()}
    _call("PUT", f"/collections/{name}", {"vectors": config})


def list_collections() -> list[str]:
    return [c["name"] for c in _call("GET", "/collections")["result"]["collections"]]


def vector_sizes(name: str) -> dict[str, int]:
    """{vector name: dims} of an existing collection."""
    vectors = _call("GET", f"/collections/{name}")["result"]["config"]["params"]["vectors"]
    return {vector: config["size"] for vector, config in vectors.items()}


def upsert(name: str, points: list[dict]) -> None:
    """Insert or replace whole points: [{"id": int, "vector": {vector_name: [floats]}, "payload": {...}}].
    A point may have no vectors yet ("vector": {})."""
    _call("PUT", f"/collections/{name}/points?wait=true", {"points": points})


def update_vectors(name: str, points: list[dict]) -> None:
    """Set some named vectors of existing points, keeping their other vectors and payload:
    [{"id": int, "vector": {vector_name: [floats]}}]."""
    _call("PUT", f"/collections/{name}/points/vectors?wait=true", {"points": points})


def count(name: str) -> int:
    return _call("POST", f"/collections/{name}/points/count", {"exact": True})["result"]["count"]


def search(name: str, vector_name: str, vector: list[float], limit: int, with_payload: bool = True) -> list[dict]:
    """Nearest chunks for a query vector: [{"id", "score", "payload"}], best first."""
    body = {"query": vector, "using": vector_name, "limit": limit, "with_payload": with_payload}
    return _call("POST", f"/collections/{name}/points/query", body)["result"]["points"]
