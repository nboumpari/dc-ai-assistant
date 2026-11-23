"""Evaluate the data agent on the golden set as a LangSmith experiment.

Run from the project root:

    uv run python -m evals.run_eval --label v2               # dev split (use while improving the prompt)
    uv run python -m evals.run_eval --label v2 --split test  # test split (final score only)

The label names the version of the prompt (prompts/system.md and the semantic
layer) and appears in the experiment name, e.g. data-agent-v2-dev-1a2b3c4d.

Scoring looks at results only, never at the wording of the answer:

  correct         questions to answer: the agent's final SQL is re-run and its
                  result compared with the expected result (see compare.py).
                  Questions with nothing to query (clarify / refuse / decline):
                  the agent's behavior field must match.
  behavior_match  the agent's behavior field matches the expected one (all questions).
"""

import argparse
import hashlib
import sys
from collections import defaultdict

from langsmith import Client, evaluate

from dc_ai_assistant import settings
from dc_ai_assistant.data_assistant.agent import PROMPT_PATH, ask, build_data_agent
from dc_ai_assistant.data_assistant.semantic_layer import SEMANTIC_LAYER_PATH
from dc_ai_assistant.data_assistant.sql_tool import SQLTool
from evals.compare import load_aliases, results_match
from evals.upload_golden_set import DATASET_NAME

_checker = SQLTool(max_rows=1000)
_aliases = load_aliases(_checker)


def correct(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["behavior"]
    if expected != "answer":
        ok = outputs.get("behavior") == expected
        return {"key": "correct", "score": int(ok), "comment": f"expected {expected}, got {outputs.get('behavior')}"}

    if outputs.get("behavior") != "answer":
        return {"key": "correct", "score": 0, "comment": f"expected an answer, got {outputs.get('behavior')}"}
    if not outputs.get("sql"):
        return {"key": "correct", "score": 0, "comment": "answered without a query"}

    result = _checker.run(outputs["sql"])
    if not result.ok:
        return {"key": "correct", "score": 0, "comment": f"agent SQL failed: {result.error}"}
    ok, reason = results_match(reference_outputs["compare"], reference_outputs["expected_rows"], result.rows, _aliases)
    return {"key": "correct", "score": int(ok), "comment": reason or "result matches"}


def behavior_match(outputs: dict, reference_outputs: dict) -> dict:
    ok = outputs.get("behavior") == reference_outputs["behavior"]
    return {"key": "behavior_match", "score": int(ok)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="version of the prompt, e.g. v4")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--repetitions", type=int, default=3, help="runs per question (answers vary between runs)")
    args = parser.parse_args()

    agent = build_data_agent()

    def target(inputs: dict) -> dict:
        result = ask(inputs["question"], agent=agent)
        return {k: v for k, v in result.items() if k != "messages"}

    client = Client()
    examples = list(client.list_examples(dataset_name=DATASET_NAME, splits=[args.split]))
    # fingerprint of everything the model is told: the prompt template and the semantic layer
    prompt_version = hashlib.sha256(PROMPT_PATH.read_bytes() + SEMANTIC_LAYER_PATH.read_bytes()).hexdigest()[:8]

    results = evaluate(
        target,
        data=examples,
        evaluators=[correct, behavior_match],
        experiment_prefix=f"data-agent-{args.label}-{args.split}",
        description=(
            f"Data agent {args.label} on the {args.split} split, model {settings.OPENAI_MODEL}, "
            f"reasoning {settings.OPENAI_REASONING_EFFORT}, temperature {settings.EFFECTIVE_TEMPERATURE}, "
            f"prompt {prompt_version}"
        ),
        metadata={
            "model": settings.OPENAI_MODEL,
            "temperature": settings.EFFECTIVE_TEMPERATURE,
            "reasoning_effort": settings.OPENAI_REASONING_EFFORT,
            "prompt_version": prompt_version,
            "version": args.label,
            "split": args.split,
            "repetitions": args.repetitions,
        },
        num_repetitions=args.repetitions,
        max_concurrency=8,
        client=client,
    )

    # one row per question per repetition
    per_question: dict[str, list[int]] = defaultdict(list)
    stratum_of: dict[str, str] = {}
    last_comment: dict[str, str] = {}
    for row in results:
        meta = row["example"].metadata
        correct_result = next(r for r in row["evaluation_results"]["results"] if r.key == "correct")
        per_question[meta["golden_id"]].append(int(correct_result.score or 0))
        stratum_of[meta["golden_id"]] = meta["stratum"]
        if not correct_result.score:
            last_comment[meta["golden_id"]] = correct_result.comment or ""

    runs = [s for scores in per_question.values() for s in scores]
    print(f"\nExperiment: {results.experiment_name}")
    print(
        f"Correct: {sum(runs)}/{len(runs)} runs = {100 * sum(runs) / len(runs):.1f}% "
        f"({len(per_question)} questions x {args.repetitions} repetitions)\n"
    )
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for golden_id, scores in per_question.items():
        by_stratum[stratum_of[golden_id]] += scores
    for stratum, scores in sorted(by_stratum.items()):
        print(f"  {stratum:<14} {sum(scores)}/{len(scores)}")
    unstable = {q: s for q, s in per_question.items() if sum(s) < len(s)}
    if unstable:
        print("\nNot always correct (passed / runs):")
        for golden_id, scores in sorted(unstable.items()):
            print(f"  {golden_id}: {sum(scores)}/{len(scores)}  last failure: {last_comment.get(golden_id, '')[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
