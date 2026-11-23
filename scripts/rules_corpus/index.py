"""Embed the chunk sets with both embedding models and load them into Qdrant.

Run inside Docker, after chunk and with Qdrant running:

    docker compose up -d qdrant
    docker compose run --rm app python -m scripts.rules_corpus.index

One Qdrant collection per chunking strategy (rules_<strategy>), one named
vector per embedding model. Each collection is created first with every chunk
as a point (text and metadata as payload, no vectors yet); vectors are then
written batch by batch as they are embedded, so the collections fill up live.

The vectors are also saved to data/index/embeddings/<strategy>.<embedder>.npy
(after every batch while running), so an interrupted run continues where it
stopped and finished files are reused (delete them to re-embed). On start, the
vectors already saved are written to Qdrant again, keeping both in step.

What is embedded is the chunk's context line (document > section) followed by
its text, the same passage the rerankers and the answering model will see.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from dc_ai_assistant.rules_assistant import models, vector_store

ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "data" / "documents" / "chunks"
EMBEDDING_DIR = ROOT / "data" / "index" / "embeddings"
STRATEGIES = ["fixed", "recursive", "structure"]
EMBEDDERS = ["gemma", "granite"]
BATCH = 64  # chunks embedded, saved and written to Qdrant at a time


def passage(chunk: dict) -> str:
    return f"{chunk['context']}\n\n{chunk['text']}"


def load_chunks(strategy: str) -> list[dict]:
    chunks = [json.loads(line) for line in (CHUNK_DIR / f"{strategy}.jsonl").read_text(encoding="utf-8").splitlines()]
    ids_path = EMBEDDING_DIR / f"{strategy}.ids.json"
    ids = [c["chunk_id"] for c in chunks]
    if ids_path.exists() and json.loads(ids_path.read_text(encoding="utf-8")) != ids:
        raise SystemExit(f"{strategy}: the chunks changed since embedding; delete its files in {EMBEDDING_DIR}")
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    return chunks


def paths(strategy: str, embedder: str) -> tuple[Path, Path]:
    return EMBEDDING_DIR / f"{strategy}.{embedder}.npy", EMBEDDING_DIR / f"{strategy}.{embedder}.partial.npy"


def saved_vectors(strategy: str, embedder: str) -> np.ndarray | None:
    done_path, partial_path = paths(strategy, embedder)
    for path in (done_path, partial_path):
        if path.exists():
            return np.load(path)
    return None


def write_vectors(collection: str, embedder: str, start: int, vectors: np.ndarray) -> None:
    points = [{"id": start + k, "vector": {embedder: v.tolist()}} for k, v in enumerate(vectors)]
    vector_store.update_vectors(collection, points)


def prepare_collection(strategy: str, chunks: list[dict]) -> str:
    """Create the collection with every chunk as a point (or reuse it), then write the vectors saved so far."""
    name = vector_store.collection_name(strategy)
    sizes = {e: models.EMBEDDERS[e].dims for e in EMBEDDERS}
    reuse = (
        name in vector_store.list_collections()
        and vector_store.vector_sizes(name) == sizes
        and vector_store.count(name) == len(chunks)
    )
    if not reuse:
        vector_store.recreate_collection(name, sizes)
        for i in range(0, len(chunks), BATCH):
            batch = range(i, min(i + BATCH, len(chunks)))
            vector_store.upsert(name, [{"id": j, "vector": {}, "payload": chunks[j]} for j in batch])
    for embedder in EMBEDDERS:
        vectors = saved_vectors(strategy, embedder)
        if vectors is not None:
            for i in range(0, len(vectors), BATCH):
                write_vectors(name, embedder, i, vectors[i : i + BATCH])
    print(f"  {strategy:9} collection '{name}': {len(chunks)} chunks", flush=True)
    return name


def embed(strategy: str, embedder: str, chunks: list[dict], collection: str) -> None:
    done_path, partial_path = paths(strategy, embedder)
    if done_path.exists():
        print(f"  {strategy:9} {embedder:5} already embedded")
        return
    vectors = saved_vectors(strategy, embedder)
    if vectors is None:
        vectors = np.zeros((0, models.EMBEDDERS[embedder].dims), dtype=np.float32)
    start, begun_at = len(vectors), time.perf_counter()
    for i in range(start, len(chunks), BATCH):
        new = models.embed_documents(embedder, [passage(c) for c in chunks[i : i + BATCH]]).astype(np.float32)
        vectors = np.vstack([vectors, new])
        np.save(partial_path, vectors)
        write_vectors(collection, embedder, i, new)
        rate = (time.perf_counter() - begun_at) / (len(vectors) - start)
        print(
            f"  {strategy:9} {embedder:5} {len(vectors):4}/{len(chunks)}  "
            f"{rate:4.1f}s per chunk  about {rate * (len(chunks) - len(vectors)) / 60:4.0f} min left",
            flush=True,
        )
    np.save(done_path, vectors)
    partial_path.unlink(missing_ok=True)


def main() -> int:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    strategies = sys.argv[1:] or STRATEGIES
    chunks = {s: load_chunks(s) for s in strategies}
    collections = {s: prepare_collection(s, chunks[s]) for s in strategies}
    for embedder in EMBEDDERS:
        for strategy in strategies:
            embed(strategy, embedder, chunks[strategy], collections[strategy])
    return 0


if __name__ == "__main__":
    sys.exit(main())
