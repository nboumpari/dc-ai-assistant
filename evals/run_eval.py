"""Evaluate the data agent on the golden set as a LangSmith experiment.

Run from the project root:

    uv run python -m evals.run_eval              # dev split (use while improving the prompt)
    uv run python -m evals.run_eval --split test # test split (final score only)

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
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    agent = build_data_agent()

    def target(inputs: dict) -> dict:
        result = ask(inputs["question"], agent=agent)
        return {k: v for k, v in result.items() if k != "messages"}

    client = Client()
    examples = list(client.list_examples(dataset_name=DATASET_NAME, splits=[args.split]))
    prompt_version = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()[:8]

    results = evaluate(
        target,
        data=examples,
        evaluators=[correct, behavior_match],
        experiment_prefix=f"data-agent-{args.split}",
        description=(
            f"Data agent on the {args.split} split, model {settings.OPENAI_MODEL}, "
            f"temperature {settings.OPENAI_TEMPERATURE}, prompt {prompt_version}"
        ),
        metadata={
            "model": settings.OPENAI_MODEL,
            "temperature": settings.OPENAI_TEMPERATURE,
            "reasoning_effort": settings.OPENAI_REASONING_EFFORT,
            "prompt_version": prompt_version,
            "split": args.split,
        },
        max_concurrency=4,
        client=client,
    )

    by_stratum: dict[str, list[int]] = defaultdict(list)
    failures = []
    for row in results:
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        meta = row["example"].metadata
        by_stratum[meta["stratum"]].append(scores.get("correct", 0))
        if not scores.get("correct"):
            comment = next((r.comment for r in row["evaluation_results"]["results"] if r.key == "correct"), "")
            failures.append((meta["golden_id"], comment))

    total = [s for scores in by_stratum.values() for s in scores]
    print(f"\nExperiment: {results.experiment_name}")
    print(f"Correct: {sum(total)}/{len(total)} = {100 * sum(total) / len(total):.0f}%\n")
    for stratum, scores in sorted(by_stratum.items()):
        print(f"  {stratum:<14} {sum(scores)}/{len(scores)}")
    if failures:
        print("\nFailed:")
        for golden_id, comment in sorted(failures):
            print(f"  {golden_id}: {comment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
