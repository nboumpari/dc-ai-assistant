"""Retrieval for the rules assistant: BM25, vector search, hybrid search and reranking.

A RetrievalConfig names one setup:
  chunking  fixed | recursive | structure      which chunk set (one Qdrant collection each)
  search    bm25 | vector | hybrid             hybrid fuses the BM25 and vector lists (RRF)
  embedder  gemma | granite                    for vector and hybrid search; None for BM25
  reranker  gte | minilm | None                reorders the candidates with a cross-encoder

First-stage search returns CANDIDATES chunks; the reranker (if any) reorders
them, and the top TOP_K are returned. BM25 runs in this process over the same
passages that were embedded (context line + chunk text).
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from dc_ai_assistant.rules_assistant import models, vector_store

ROOT = Path(__file__).resolve().parents[3]
CHUNK_DIR = ROOT / "data" / "documents" / "chunks"

CHUNKINGS = ["fixed", "recursive", "structure"]
CANDIDATES = 20  # chunks from each first-stage search, and the reranker's input
TOP_K = 5  # chunks returned
RRF_K = 60  # reciprocal rank fusion constant (the usual default)

TOKEN = re.compile(r"\w+(?:[.\-/]\w+)*")  # keeps "1.1.3.6", "e-mail" and "UN/ECE" as single tokens


@dataclass(frozen=True)
class RetrievalConfig:
    chunking: str
    search: str
    embedder: str | None = None
    reranker: str | None = None

    @property
    def name(self) -> str:
        parts = [self.chunking, self.search] + ([self.embedder] if self.embedder else [])
        return "-".join(parts + [self.reranker or "norerank"])


def all_configs() -> list[RetrievalConfig]:
    """Every combination: 3 chunkings x (BM25 + 2 embedders x {vector, hybrid}) x 3 reranker options = 45."""
    searches = [("bm25", None)] + [(s, e) for e in models.EMBEDDERS for s in ("vector", "hybrid")]
    rerankers = [None, *models.RERANKERS]
    return [RetrievalConfig(c, s, e, r) for c in CHUNKINGS for s, e in searches for r in rerankers]


def passage(chunk: dict) -> str:
    """What is embedded, searched and shown to rerankers: the context line, then the chunk text."""
    return f"{chunk['context']}\n\n{chunk['text']}"


@cache
def load_chunks(chunking: str) -> list[dict]:
    lines = (CHUNK_DIR / f"{chunking}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@cache
def _bm25(chunking: str) -> BM25Okapi:
    return BM25Okapi([tokenize(passage(c)) for c in load_chunks(chunking)])


def bm25_search(chunking: str, question: str, limit: int = CANDIDATES) -> list[int]:
    """Indices of the best-matching chunks by BM25, best first."""
    scores = _bm25(chunking).get_scores(tokenize(question))
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]


def vector_search(chunking: str, embedder: str, query_vector, limit: int = CANDIDATES) -> list[int]:
    """Indices of the nearest chunks in Qdrant (point id = chunk index), best first."""
    hits = vector_store.search(
        vector_store.collection_name(chunking), embedder, list(map(float, query_vector)), limit, with_payload=False
    )
    return [hit["id"] for hit in hits]


def rrf(rankings: list[list[int]], k: int = RRF_K, limit: int = CANDIDATES) -> list[int]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to an item's score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]


def first_stage(config: RetrievalConfig, question: str, query_vector=None) -> list[int]:
    """The CANDIDATES chunk indices before reranking."""
    if config.search == "bm25":
        return bm25_search(config.chunking, question)
    if query_vector is None:
        query_vector = models.embed_query(config.embedder, question)
    vector = vector_search(config.chunking, config.embedder, query_vector)
    if config.search == "vector":
        return vector
    return rrf([bm25_search(config.chunking, question), vector])


def rerank(reranker: str, question: str, chunking: str, candidates: list[int]) -> list[int]:
    chunks = load_chunks(chunking)
    scores = models.rerank_scores(reranker, question, [passage(chunks[i]) for i in candidates])
    order = sorted(range(len(candidates)), key=lambda j: -scores[j])
    return [candidates[j] for j in order]


def retrieve(config: RetrievalConfig, question: str) -> list[dict]:
    """The TOP_K chunks for a question, best first."""
    candidates = first_stage(config, question)
    ranked = rerank(config.reranker, question, config.chunking, candidates) if config.reranker else candidates
    chunks = load_chunks(config.chunking)
    return [chunks[i] for i in ranked[:TOP_K]]
