"""LLM judge for the rules agent's saved evaluation runs (Google Gemini).

Run after run_rag_eval, on the saved runs (no local models needed):

    uv run python -m evals.rules.judge --label v1              # dev split
    uv run python -m evals.rules.judge --label v1 --split test

The judge is from a different model family than the agent's OpenAI model, so
it does not favour answers in its own style. It sees the question, the
reference answer, the passages the agent was given and the agent's answer, and
gives two verdicts with its reasoning first (see judge_prompt.md):

  correctness   correct / partially_correct / incorrect   (against the reference answer)
  groundedness  grounded / partly_grounded / not_grounded (against the passages)

Only questions that should be answered are judged. If the agent said not found
or declined such a question, its answer is incorrect without asking the judge.

The headline score per run, correct, is:
  questions to answer      the judge's verdict is "correct"
  not found and decline    the agent's behaviour matches (scored by code in run_rag_eval)

Verdicts are saved to data/eval_runs/rules-agent-<label>-<split>.judge.jsonl,
one line per run, so judging continues where it stopped.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from dc_ai_assistant import settings  # noqa: F401  (loads .env, e.g. GEMINI_API_KEY)
from dc_ai_assistant.rules_assistant import retrieval
from dc_ai_assistant.rules_assistant.agent import format_passages
from evals.rules.run_rag_eval import load_runs, runs_path

JUDGE_MODEL = "gemini-3.8-flash"
JUDGE_PROMPT = Path(__file__).with_name("judge_prompt.md")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
WORKERS = 4
RETRIES = 5

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "correctness_reasoning": {"type": "STRING"},
        "correctness": {"type": "STRING", "enum": ["correct", "partially_correct", "incorrect"]},
        "groundedness_reasoning": {"type": "STRING"},
        "groundedness": {"type": "STRING", "enum": ["grounded", "partly_grounded", "not_grounded"]},
    },
    "required": ["correctness_reasoning", "correctness", "groundedness_reasoning", "groundedness"],
    "propertyOrdering": ["correctness_reasoning", "correctness", "groundedness_reasoning", "groundedness"],
}


def judge_path(label: str, split: str) -> Path:
    return runs_path(label, split).with_suffix(".judge.jsonl")


def judge_input(run: dict) -> str:
    chunks = {c["chunk_id"]: c for chunking in retrieval.CHUNKINGS for c in retrieval.load_chunks(chunking)}
    passages = [chunks[p["chunk_id"]] for p in run["output"]["passages"]]
    return (
        f"Question:\n{run['question']}\n\n"
        f"Reference answer:\n{run['reference']['reference_answer']}\n\n"
        f"Passages the assistant was given:\n\n{format_passages(passages)}\n\n"
        f"Assistant's answer:\n{run['output']['answer']}"
    )


def ask_judge(text: str) -> dict:
    body = {
        "systemInstruction": {"parts": [{"text": JUDGE_PROMPT.read_text(encoding="utf-8")}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": VERDICT_SCHEMA,
        },
    }
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    for attempt in range(RETRIES):
        response = requests.post(API_URL.format(model=JUDGE_MODEL), json=body, headers=headers, timeout=120)
        if response.status_code in (429, 500, 503) and attempt < RETRIES - 1:
            time.sleep(2**attempt * 5)  # rate limit or overload: wait and retry
            continue
        response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        return json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
    raise RuntimeError("the judge did not answer")


def verdict(run: dict) -> dict:
    record = {"golden_id": run["golden_id"], "repetition": run["repetition"], "judge_model": JUDGE_MODEL}
    if run["output"]["behavior"] != "answer":
        reason = f"The assistant gave no answer ({run['output']['behavior']}) to a question that has one."
        return {**record, "correctness_reasoning": reason, "correctness": "incorrect", "groundedness": None}
    return {**record, **ask_judge(judge_input(run))}


def judge_runs(label: str, split: str) -> list[dict]:
    runs = [r for r in load_runs(runs_path(label, split)) if r["reference"]["behavior"] == "answer"]
    path = judge_path(label, split)
    done = load_runs(path)
    finished = {(v["golden_id"], v["repetition"]) for v in done}
    todo = [r for r in runs if (r["golden_id"], r["repetition"]) not in finished]
    print(f"{len(runs)} answerable runs, {len(todo)} to judge with {JUDGE_MODEL}", flush=True)
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as pool:
        for n, result in enumerate(pool.map(verdict, todo), start=1):
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"  {n:3}/{len(todo)} {result['golden_id']} run {result['repetition']}: "
                f"{result['correctness']}, {result['groundedness']}",
                flush=True,
            )
    return load_runs(path)


def report(label: str, split: str) -> None:
    runs = load_runs(runs_path(label, split))
    verdicts = {(v["golden_id"], v["repetition"]): v for v in load_runs(judge_path(label, split))}
    per_question: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        if r["reference"]["behavior"] == "answer":
            v = verdicts.get((r["golden_id"], r["repetition"]))
            if v is None:
                continue
            ok = int(v["correctness"] == "correct")
        else:
            ok = r["scores"]["behavior_match"]
        per_question[r["golden_id"]].append(ok)

    scored = [s for scores in per_question.values() for s in scores]
    print(f"\n{label} on {split}: correct {sum(scored)}/{len(scored)} runs = {100 * sum(scored) / len(scored):.1f}%")
    judged = list(verdicts.values())
    print("  correctness  ", dict(Counter(v["correctness"] for v in judged)))
    print("  groundedness ", dict(Counter(v["groundedness"] for v in judged if v["groundedness"])))
    failing = {q: s for q, s in sorted(per_question.items()) if sum(s) < len(s)}
    if failing:
        print("\n  Not always correct (passed / runs):")
        for gid, scores in failing.items():
            notes = [f"{v['correctness']}/{v['groundedness']}" for (g, _), v in sorted(verdicts.items()) if g == gid]
            print(f"    {gid}: {sum(scores)}/{len(scores)}  {', '.join(notes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    judge_runs(args.label, args.split)
    report(args.label, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
