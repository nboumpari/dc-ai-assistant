"""Evaluate the supervisor's router: does every message go to the right place?

Run from the project root (only the router runs, no agent, so no Docker needed):

    uv run python -m evals.supervisor.run_router_eval --label v1              # dev split
    uv run python -m evals.supervisor.run_router_eval --label v1 --split test # final score only

The label names the version of the router prompt (supervisor/prompts/router.md).

Questions and their expected route come from three files:
  evals/golden_set.yaml              the data agent's golden set
  evals/rules/golden_set.yaml        the rules agent's golden set
  evals/supervisor/golden_set.yaml   extra fallback questions added for this evaluation
The golden sets' labels map to the router's routes: clarify -> data_agent (the
data agent asks the clarifying question), out_of_scope -> fallback.

Each question is sent on its own (no earlier conversation), --repetitions
times. Scored by code: routing accuracy, per expected route, and a confusion
table. Runs are saved to data/eval_runs/router-<label>-<split>.jsonl as they
finish, so an interrupted evaluation continues where it stopped.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from dc_ai_assistant.supervisor.agent import PROMPT_PATH, route_question

EVALS = Path(__file__).resolve().parents[1]
SOURCES = {
    "data": EVALS / "golden_set.yaml",
    "rules": EVALS / "rules" / "golden_set.yaml",
    "added": EVALS / "supervisor" / "golden_set.yaml",
}
RUNS_DIR = EVALS.parent / "data" / "eval_runs"
ROUTE_OF_LABEL = {"clarify": "data_agent", "out_of_scope": "fallback"}
ROUTES = ["data_agent", "rules_agent", "fallback"]
WORKERS = 8


def load_questions(split: str) -> list[dict]:
    questions = []
    for source, path in SOURCES.items():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = data["questions"] if isinstance(data, dict) else data
        for q in items:
            if q["split"] == split:
                route = ROUTE_OF_LABEL.get(q["route"], q["route"])
                questions.append({"id": q["id"], "question": q["question"], "expected": route, "source": source})
    return questions


def prompt_version() -> str:
    return hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:8]


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(label: str, split: str, repetitions: int) -> Path:
    path = RUNS_DIR / f"router-{label}-{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_runs(path)
    version = prompt_version()
    if done and done[0]["prompt_version"] != version:
        raise SystemExit(f"{path.name} was made with another prompt ({done[0]['prompt_version']}); use a new label")
    finished = {(r["id"], r["repetition"]) for r in done}
    questions = load_questions(split)
    todo = [(q, rep) for rep in range(1, repetitions + 1) for q in questions if (q["id"], rep) not in finished]
    print(f"{len(questions)} {split} questions x {repetitions} repetitions: {len(todo)} runs to do", flush=True)

    def route(item):
        q, rep = item
        try:
            decision = route_question(q["question"])
        except Exception as error:  # an API failure: not saved, so a rerun retries it
            return None, f"{q['id']} run {rep}: failed ({type(error).__name__})"
        record = {
            **q,
            "repetition": rep,
            "route": decision.route,
            "reason": decision.reason,
            "correct": int(decision.route == q["expected"]),
            "prompt_version": version,
            "label": label,
            "split": split,
        }
        return record, None

    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as pool:
        for record, failure in pool.map(route, todo):
            if failure:
                print(f"  {failure}", flush=True)
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return path


def report(path: Path) -> None:
    runs = load_runs(path)
    correct = sum(r["correct"] for r in runs)
    print(f"\n{path.name}: routing accuracy {correct}/{len(runs)} runs = {100 * correct / len(runs):.1f}%")
    by_expected: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        by_expected[r["expected"]].append(r["correct"])
    for route in ROUTES:
        scores = by_expected.get(route, [])
        if scores:
            print(f"  expected {route:12} {sum(scores)}/{len(scores)}")

    confusion = Counter((r["expected"], r["route"]) for r in runs)
    print(f"\n  {'expected \\ routed to':22}" + "".join(f"{r:>13}" for r in ROUTES))
    for expected in ROUTES:
        print(f"  {expected:22}" + "".join(f"{confusion.get((expected, r), 0):>13}" for r in ROUTES))

    per_question: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        per_question[r["id"]].append(r)
    wrong = {q: rs for q, rs in sorted(per_question.items()) if not all(r["correct"] for r in rs)}
    if wrong:
        print("\n  Misrouted (correct runs / runs):")
        for qid, rs in wrong.items():
            routes = [r["route"] for r in rs if not r["correct"]]
            print(
                f"    {qid} ({rs[0]['source']}): {sum(r['correct'] for r in rs)}/{len(rs)} "
                f"expected {rs[0]['expected']}, got {routes} | {rs[0]['question'][:70]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="version of the router prompt, e.g. v3")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--repetitions", type=int, default=3, help="runs per question (the route can vary)")
    args = parser.parse_args()
    report(run(args.label, args.split, args.repetitions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
