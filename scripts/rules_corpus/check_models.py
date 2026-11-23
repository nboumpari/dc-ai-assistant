"""Smoke test for the local embedding models and rerankers.

Run inside Docker:

    docker compose run --rm app python -m scripts.rules_corpus.check_models

For a few dev questions, takes the structure chunk that holds the evidence
plus some chunks from other documents, and checks that every embedder and
reranker ranks the evidence chunk first. Also prints load and run times.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from dc_ai_assistant.rules_assistant import models
from dc_ai_assistant.rules_assistant.retrieval import passage

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "documents" / "chunks" / "structure.jsonl"
GOLDEN = ROOT / "evals" / "rules" / "golden_set.yaml"
QUESTION_IDS = ["R04", "R14", "R18", "R39", "R44"]
DISTRACTORS = 7


def main() -> int:
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    golden = {q["id"]: q for q in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["questions"]}
    rng = random.Random(0)
    cases = []
    for qid in QUESTION_IDS:
        q = golden[qid]
        span = q["evidence"][0]
        target = next(
            c
            for c in chunks
            if c["document"] == q["document"] and c["char_start"] <= span["char_start"] < c["char_end"]
        )
        others = rng.sample([c for c in chunks if c["document"] != q["document"]], DISTRACTORS)
        cases.append((qid, q["question"], [passage(target)] + [passage(c) for c in others]))

    failures = 0
    for name in models.EMBEDDERS:
        start = time.perf_counter()
        try:
            models.embed_query(name, "warm-up")
        except Exception as error:  # e.g. a gated model without access
            print(f"embedder {name:6} SKIPPED: {type(error).__name__}: {str(error).splitlines()[0][:120]}")
            continue
        load = time.perf_counter() - start
        start = time.perf_counter()
        ranks = []
        for _, question, passages in cases:
            docs = models.embed_documents(name, passages)
            scores = docs @ models.embed_query(name, question)
            ranks.append(int((scores > scores[0]).sum()) + 1)
        run = (time.perf_counter() - start) / len(cases)
        failures += sum(r != 1 for r in ranks)
        dims = docs.shape[1]
        print(f"embedder {name:6} dims {dims:5}  load {load:5.1f}s  {run:5.2f}s per question  evidence ranks {ranks}")

    for name in models.RERANKERS:
        start = time.perf_counter()
        models.rerank_scores(name, "warm-up", ["warm-up"])
        load = time.perf_counter() - start
        start = time.perf_counter()
        ranks = []
        for _, question, passages in cases:
            scores = np.array(models.rerank_scores(name, question, passages))
            ranks.append(int((scores > scores[0]).sum()) + 1)
        run = (time.perf_counter() - start) / len(cases)
        failures += sum(r != 1 for r in ranks)
        print(
            f"reranker {name:6}             load {load:5.1f}s  {run:5.2f}s per {len(passages)} passages  "
            f"evidence ranks {ranks}"
        )

    print("all evidence chunks ranked first" if not failures else f"{failures} evidence chunk(s) not ranked first")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
