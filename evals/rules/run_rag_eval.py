"""Evaluate the rules agent (advanced RAG) on the golden set.

Run the agent inside Docker (it needs the local models and Qdrant):

    docker compose run --rm app python -m evals.rules.run_rag_eval --label v1              # dev split
    docker compose run --rm app python -m evals.rules.run_rag_eval --label v1 --split test # final score only

The label names the prompt version (see src/dc_ai_assistant/rules_assistant/prompts/CHANGELOG.md).
Every question is run --repetitions times, because answers vary between runs.
Each run is saved to data/eval_runs/rules-agent-<label>-<split>.jsonl as soon as
it finishes, so an interrupted evaluation continues where it stopped. A file
belongs to one version of the prompts; changing a prompt needs a new label.

Scored by code, for every run:

  behavior_match   the agent answered / said not found / declined as expected (all questions)
  citation_hit     a cited passage overlaps the evidence (questions to answer)
  coverage@5       share of the evidence in the 5 passages the agent was given (questions to answer)
  latency_s        seconds for the whole question

Correctness and groundedness of the answers are judged by an LLM from a
different model family in a separate step, on the saved runs.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from langsmith import Client

from dc_ai_assistant.rules_assistant.agent import (
    ANSWER_PROMPT,
    REFORMULATE_PROMPT,
    RETRIEVAL,
    ask,
    build_rules_agent,
)
from dc_ai_assistant.rules_assistant.retrieval import RetrievalConfig, all_configs
from evals.rules.retrieval_metrics import coverage_at, hit_at
from evals.rules.upload_golden_set import DATASET_NAME

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "eval_runs"


def prompt_version() -> str:
    """Fingerprint of everything the model is told: both prompt files."""
    return hashlib.sha256(REFORMULATE_PROMPT.read_bytes() + ANSWER_PROMPT.read_bytes()).hexdigest()[:8]


def runs_path(label: str, split: str) -> Path:
    return RUNS_DIR / f"rules-agent-{label}-{split}.jsonl"


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def code_scores(output: dict, reference: dict) -> dict:
    scores = {"behavior_match": int(output["behavior"] == reference["behavior"])}
    if reference["behavior"] == "answer":
        evidence = reference["evidence"]
        cited = output.get("cited_chunks", [])
        scores["citation_hit"] = hit_at(cited, evidence, len(cited)) if cited else 0
        scores["coverage@5"] = coverage_at(output.get("passages", []), evidence, 5)
    return scores


def run(label: str, split: str, repetitions: int, config: RetrievalConfig = RETRIEVAL) -> Path:
    examples = sorted(
        Client().list_examples(dataset_name=DATASET_NAME, splits=[split]), key=lambda e: e.metadata["golden_id"]
    )
    if config != RETRIEVAL:  # runs with another retriever are saved under their own name
        label = f"{label}-{config.name}"
    path = runs_path(label, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_runs(path)
    version = prompt_version()
    if done and done[0]["prompt_version"] != version:
        raise SystemExit(f"{path.name} was made with other prompts ({done[0]['prompt_version']}); use a new label")
    finished = {(r["golden_id"], r["repetition"]) for r in done}

    agent = build_rules_agent(config=config)
    todo = [(e, rep) for rep in range(1, repetitions + 1) for e in examples]
    todo = [(e, rep) for e, rep in todo if (e.metadata["golden_id"], rep) not in finished]
    print(f"{len(examples)} {split} questions x {repetitions} repetitions: {len(todo)} runs to do", flush=True)
    with path.open("a", encoding="utf-8") as out:
        for n, (example, rep) in enumerate(todo, start=1):
            gid, question = example.metadata["golden_id"], example.inputs["question"]
            start = time.perf_counter()
            try:
                output = ask(question, agent=agent)
            except Exception as error:  # an API failure: not saved, so a rerun retries it
                print(f"  {n:3}/{len(todo)} {gid} run {rep}: failed ({type(error).__name__}: {error})", flush=True)
                continue
            record = {
                "golden_id": gid,
                "repetition": rep,
                "question": question,
                "reference": example.outputs,
                "metadata": example.metadata,
                "output": output,
                "latency_s": round(time.perf_counter() - start, 2),
                "scores": code_scores(output, example.outputs),
                "prompt_version": version,
                "retrieval": config.name,
                "label": label,
                "split": split,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            ok = "ok" if record["scores"]["behavior_match"] else "BEHAVIOR"
            seconds = record["latency_s"]
            print(f"  {n:3}/{len(todo)} {gid} run {rep}: {output['behavior']:9} {ok:8} {seconds:5.1f}s", flush=True)
    return path


def summarise(path: Path) -> None:
    runs = load_runs(path)
    if not runs:
        return
    answerable = [r for r in runs if r["reference"]["behavior"] == "answer"]
    print(f"\n{path.name}: {len(runs)} runs, prompts {runs[0]['prompt_version']}")
    print(f"  behavior_match  {sum(r['scores']['behavior_match'] for r in runs)}/{len(runs)}")
    if answerable:
        cited = sum(r["scores"]["citation_hit"] for r in answerable)
        print(f"  citation_hit    {cited}/{len(answerable)} answerable runs")
        print(f"  coverage@5      {mean(r['scores']['coverage@5'] for r in answerable):.3f}")
    print(f"  latency         {mean(r['latency_s'] for r in runs):.1f}s mean")

    by_behavior: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        by_behavior[r["reference"]["behavior"]].append(r["scores"]["behavior_match"])
    for behavior, scores in sorted(by_behavior.items()):
        print(f"    expected {behavior:9} {sum(scores)}/{len(scores)}")

    per_question: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        per_question[r["golden_id"]].append(r)
    issues = []
    for gid, rs in sorted(per_question.items()):
        wrong = [r["output"]["behavior"] for r in rs if not r["scores"]["behavior_match"]]
        uncited = [r for r in rs if r["reference"]["behavior"] == "answer" and not r["scores"]["citation_hit"]]
        if wrong or uncited:
            detail = f"behavior {rs[0]['reference']['behavior']} got {wrong}" if wrong else ""
            if uncited:
                detail += f"{'; ' if detail else ''}evidence not cited in {len(uncited)}/{len(rs)} runs"
                detail += f" (coverage@5 {mean(r['scores']['coverage@5'] for r in rs):.2f})"
            issues.append(f"    {gid}: {detail}")
    if issues:
        print("\n  Questions with code-scored issues:")
        print("\n".join(issues))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="prompt version, e.g. v1 (see prompts/CHANGELOG.md)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--repetitions", type=int, default=3, help="runs per question (answers vary between runs)")
    parser.add_argument(
        "--retrieval",
        default=RETRIEVAL.name,
        help=f"retrieval config name from the benchmark (default: the agent's own, {RETRIEVAL.name})",
    )
    args = parser.parse_args()
    configs = {c.name: c for c in all_configs()}
    if args.retrieval not in configs:
        raise SystemExit(f"unknown retrieval config {args.retrieval}; see evals/rules/retrieval_results.md")
    summarise(run(args.label, args.split, args.repetitions, configs[args.retrieval]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
