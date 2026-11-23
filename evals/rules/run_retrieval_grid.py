"""Retrieval benchmark: every retrieval config on the dev split, one LangSmith experiment each.

Run inside Docker, with Qdrant running and the chunks indexed:

    docker compose run --rm app python -m evals.rules.run_retrieval_grid               # all 45 configs
    docker compose run --rm app python -m evals.rules.run_retrieval_grid --limit 3 --no-upload   # smoke test

Only questions that should be answered are used (not-found and out-of-scope
questions have no evidence to find).

Retrieval runs as in the rules agent: each question is first rewritten by the
agent's reformulate step, the reformulated query is searched, and the reranker
scores the candidates against the original question. The reformulation is
made once per question and cached, so every config searches the same query;
apart from it no LLM is involved, and each config runs once. Scores per
question, averaged per config (see retrieval_metrics.py):

  hit@5, coverage@5, mrr@5   the TOP_K chunks the answering model would get
  hit@20_first_stage         is the evidence among the CANDIDATES before reranking
  latency_ms                 query embedding + search + reranking for one question
                             (the reformulation is the same for every config, so not counted)

The expensive parts are computed once and cached in data/index/grid/ (delete
it after re-chunking, changing models or changing the reformulate prompt):
reformulated queries, query embeddings per model, first-stage results per
chunking and search, and reranker scores per question and chunk (configs of a
chunking share most of their candidates). A config's latency is the sum of the
measured times of the parts it uses; reranking time is the measured time per
scored passage, summed over its candidates.

Results are also written to evals/rules/retrieval_results.md.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean

from langsmith import Client, evaluate

from dc_ai_assistant.rules_assistant import models, retrieval
from dc_ai_assistant.rules_assistant.agent import REFORMULATE_PROMPT, reformulate_question
from dc_ai_assistant.rules_assistant.retrieval import CANDIDATES, TOP_K, RetrievalConfig
from evals.rules.retrieval_metrics import coverage_at, hit_at, reciprocal_rank_at
from evals.rules.upload_golden_set import DATASET_NAME

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "index" / "grid"
RESULTS_PATH = Path(__file__).with_name("retrieval_results.md")
SPLIT = "dev"


class JsonCache:
    """A dict saved to a JSON file after every change, so long runs can resume."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data), encoding="utf-8")


def prompt_version() -> str:
    """Fingerprint of the reformulate prompt the queries were made with."""
    return hashlib.sha256(REFORMULATE_PROMPT.read_bytes()).hexdigest()[:8]


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - start) * 1000


def load_questions(client: Client, limit: int | None) -> list:
    examples = client.list_examples(dataset_name=DATASET_NAME, splits=[SPLIT])
    answerable = [e for e in examples if e.outputs["behavior"] == "answer"]
    answerable.sort(key=lambda e: e.metadata["golden_id"])
    return answerable[:limit] if limit else answerable


def compute_search_queries(questions: list) -> dict:
    """The reformulated search query per question, made once by the agent's reformulate step."""
    cache = JsonCache("reformulations")
    for q in questions:
        gid = q.metadata["golden_id"]
        if gid not in cache.data:
            result = reformulate_question(q.inputs["question"])
            cache.data[gid] = {"search_query": result.search_query, "in_scope": result.in_scope}
            cache.save()
    return {gid: r["search_query"] for gid, r in cache.data.items()}


def compute_queries(questions: list, search_queries: dict) -> dict:
    """Query embedding and its time, per embedder and question."""
    cache = JsonCache("queries")
    for embedder in models.EMBEDDERS:
        done = cache.data.setdefault(embedder, {})
        todo = [q for q in questions if q.metadata["golden_id"] not in done]
        if todo:
            models.embed_query(embedder, "warm-up")
        for q in todo:
            gid = q.metadata["golden_id"]
            vector, ms = timed(models.embed_query, embedder, search_queries[gid])
            done[gid] = {"vector": vector.tolist(), "ms": ms}
        cache.save()
    return cache.data


def compute_first_stage(questions: list, queries: dict, search_queries: dict) -> dict:
    """Candidate chunk indices and search time, per chunking, search and question (before fusion)."""
    cache = JsonCache("first_stage")
    for chunking in retrieval.CHUNKINGS:
        retrieval.bm25_search(chunking, "warm-up")
        lists = cache.data.setdefault(chunking, {})
        for q in questions:
            gid = q.metadata["golden_id"]
            if gid not in lists.setdefault("bm25", {}):
                ids, ms = timed(retrieval.bm25_search, chunking, search_queries[gid])
                lists["bm25"][gid] = {"ids": ids, "ms": ms}
            for embedder in models.EMBEDDERS:
                key = f"vector-{embedder}"
                if gid not in lists.setdefault(key, {}):
                    vector = queries[embedder][gid]["vector"]
                    ids, ms = timed(retrieval.vector_search, chunking, embedder, vector)
                    lists[key][gid] = {"ids": ids, "ms": ms}
        cache.save()
    return cache.data


def candidates_of(config: RetrievalConfig, gid: str, first: dict) -> list[int]:
    lists = first[config.chunking]
    if config.search == "bm25":
        return lists["bm25"][gid]["ids"]
    vector = lists[f"vector-{config.embedder}"][gid]["ids"]
    return vector if config.search == "vector" else retrieval.rrf([lists["bm25"][gid]["ids"], vector])


def compute_rerank_scores(questions: list, first: dict, configs: list[RetrievalConfig]) -> dict:
    """Score and time per scored passage, per reranker, chunking, question and chunk."""
    cache = JsonCache("rerank")
    for reranker in models.RERANKERS:
        models.rerank_scores(reranker, "warm-up", ["warm-up"])
        for chunking in retrieval.CHUNKINGS:
            scores = cache.data.setdefault(reranker, {}).setdefault(chunking, {})
            for n, q in enumerate(questions, start=1):
                gid, question = q.metadata["golden_id"], q.inputs["question"]
                done = scores.setdefault(gid, {})
                needed = {
                    i
                    for c in configs
                    if c.chunking == chunking and c.reranker == reranker
                    for i in candidates_of(c, gid, first)
                }
                todo = sorted(i for i in needed if str(i) not in done)
                if not todo:
                    continue
                chunks = retrieval.load_chunks(chunking)
                passages = [retrieval.passage(chunks[i]) for i in todo]
                result, ms = timed(models.rerank_scores, reranker, question, passages)
                for i, score in zip(todo, result, strict=True):
                    done[str(i)] = {"score": score, "ms": ms / len(todo)}
                cache.save()
                print(f"  rerank {reranker:6} {chunking:9} {n:2}/{len(questions)}  {len(todo)} passages", flush=True)
    return cache.data


def run_config(
    config: RetrievalConfig, questions: list, queries: dict, first: dict, rerank: dict, search_queries: dict
) -> dict:
    """Per question: the returned chunks, the first-stage chunks and the latency."""
    chunks = retrieval.load_chunks(config.chunking)
    out = {}
    for q in questions:
        gid = q.metadata["golden_id"]
        candidates = candidates_of(config, gid, first)
        latency = 0.0
        if config.search in ("bm25", "hybrid"):
            latency += first[config.chunking]["bm25"][gid]["ms"]
        if config.embedder:
            latency += queries[config.embedder][gid]["ms"]
            latency += first[config.chunking][f"vector-{config.embedder}"][gid]["ms"]
        ranked = candidates
        if config.reranker:
            scored = rerank[config.reranker][config.chunking][gid]
            ranked = sorted(candidates, key=lambda i: -scored[str(i)]["score"])
            latency += sum(scored[str(i)]["ms"] for i in candidates)

        def brief(i: int) -> dict:
            c = chunks[i]
            keys = ("chunk_id", "document", "section_id", "char_start", "char_end")
            return {k: c[k] for k in keys}

        out[gid] = {
            "chunks": [brief(i) for i in ranked[:TOP_K]],
            "first_stage": [brief(i) for i in candidates],
            "latency_ms": round(latency, 1),
            "search_query": search_queries[gid],
        }
    return out


def scores_for(outputs: dict, evidence: list[dict]) -> dict:
    return {
        "hit@5": hit_at(outputs["chunks"], evidence, TOP_K),
        "coverage@5": coverage_at(outputs["chunks"], evidence, TOP_K),
        "mrr@5": reciprocal_rank_at(outputs["chunks"], evidence, TOP_K),
        "hit@20_first_stage": hit_at(outputs["first_stage"], evidence, CANDIDATES),
        "latency_ms": outputs["latency_ms"],
    }


def retrieval_scores(outputs: dict, reference_outputs: dict) -> list[dict]:
    """LangSmith evaluator: all retrieval metrics for one question."""
    return [{"key": k, "score": v} for k, v in scores_for(outputs, reference_outputs["evidence"]).items()]


def upload(config: RetrievalConfig, questions: list, results: dict, client: Client) -> None:
    def target(inputs: dict) -> dict:
        return results[by_question[inputs["question"]]]

    by_question = {q.inputs["question"]: q.metadata["golden_id"] for q in questions}
    evaluate(
        target,
        data=questions,
        evaluators=[retrieval_scores],
        experiment_prefix=f"retrieval-{config.name}",
        description=(
            f"Retrieval on the {SPLIT} split with reformulated queries: {config.chunking} chunks, "
            f"{config.search} search{f' with {config.embedder}' if config.embedder else ''}, "
            f"{f'reranked by {config.reranker}' if config.reranker else 'no reranker'}; "
            f"{CANDIDATES} candidates, top {TOP_K} returned"
        ),
        metadata={
            "stage": "retrieval",
            "reformulate_prompt": prompt_version(),
            "chunking": config.chunking,
            "search": config.search,
            "embedder": config.embedder or "none",
            "reranker": config.reranker or "none",
            "candidates": CANDIDATES,
            "top_k": TOP_K,
            "split": SPLIT,
        },
        max_concurrency=8,
        client=client,
    )


def write_table(rows: list[tuple[RetrievalConfig, dict]], n_questions: int) -> None:
    lines = [
        "# Retrieval benchmark (dev split)",
        "",
        f"{n_questions} answerable dev questions, searched with the rules agent's reformulated query "
        f"(reformulate prompt {prompt_version()}); the reranker scores against the original question. "
        f"{CANDIDATES} candidates, top {TOP_K} returned. "
        "Sorted by coverage@5, then MRR@5. Latency: mean per question on the CPU, in milliseconds.",
        "",
        "| # | config | chunking | search | embedder | reranker "
        "| hit@5 | coverage@5 | MRR@5 | hit@20 first stage | latency ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, (c, m) in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{c.name}` | {c.chunking} | {c.search} | {c.embedder or '-'} | {c.reranker or '-'} | "
            f"{m['hit@5']:.3f} | {m['coverage@5']:.3f} | {m['mrr@5']:.3f} | {m['hit@20_first_stage']:.3f} | "
            f"{m['latency_ms']:.0f} |"
        )
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--no-upload", action="store_true", help="compute and print, create no experiments")
    args = parser.parse_args()

    client = Client()
    questions = load_questions(client, args.limit)
    configs = retrieval.all_configs()
    print(f"{len(questions)} answerable {SPLIT} questions, {len(configs)} configs", flush=True)

    search_queries = compute_search_queries(questions)
    queries = compute_queries(questions, search_queries)
    first = compute_first_stage(questions, queries, search_queries)
    rerank = compute_rerank_scores(questions, first, configs)

    rows = []
    for config in configs:
        results = run_config(config, questions, queries, first, rerank, search_queries)
        per_question = [scores_for(results[q.metadata["golden_id"]], q.outputs["evidence"]) for q in questions]
        rows.append((config, {k: mean(s[k] for s in per_question) for k in per_question[0]}))
        if not args.no_upload:
            upload(config, questions, results, client)
    rows.sort(key=lambda r: (-r[1]["coverage@5"], -r[1]["mrr@5"], r[1]["latency_ms"]))

    if not args.limit:
        write_table(rows, len(questions))
    print(f"\n{'config':42} hit@5  cov@5  mrr@5  hit@20  latency ms")
    for c, m in rows[:15]:
        print(
            f"{c.name:42} {m['hit@5']:.3f}  {m['coverage@5']:.3f}  {m['mrr@5']:.3f}  "
            f"{m['hit@20_first_stage']:.3f}  {m['latency_ms']:8.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
