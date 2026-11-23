"""Upload (or sync) the rules agent's golden set to a LangSmith dataset.

Run from the project root, after the set has been reviewed:

    uv run python -m evals.rules.upload_golden_set

Needs LANGSMITH_API_KEY in .env. Safe to run again: examples are matched by
their golden-set id, so changed questions are updated, new ones added and
removed ones deleted. The dataset keeps its history and past experiments.

In LangSmith each example has:
  inputs    {question}
  outputs   expected behaviour and route, reference answer, and the evidence
            (document, character span and quote) that retrieval should find
  metadata  golden_id, question_type, difficulty, document (filter and group by these)
  split     dev or test
"""

import sys

import yaml

from evals.rules.generate_golden_set import GOLDEN_PATH
from evals.upload_golden_set import sync_dataset

DATASET_NAME = "dc-rules-golden-set"
DATASET_DESCRIPTION = (
    "Golden set for the rules agent (RAG over GS1 standards, EU and UN transport and safety "
    "regulations, and HSE warehouse guidance). Generated from whole document sections with verbatim "
    "evidence spans, then reviewed by a person. Includes not-in-corpus and out-of-scope questions. "
    "Splits: dev (retrieval and RAG tuning) and test (final score)."
)


def to_example(q: dict) -> dict:
    return {
        "inputs": {"question": q["question"]},
        "outputs": {
            "behavior": q["behavior"],
            "route": q["route"],
            "reference_answer": q.get("reference_answer"),
            "document": q.get("document"),
            "sections": q.get("sections") or [],
            "pages": q.get("pages") or [],
            "evidence": [
                {"document": q["document"], **{k: e[k] for k in ("char_start", "char_end", "quote")}}
                for e in q.get("evidence") or []
            ],
        },
        "metadata": {
            "golden_id": q["id"],
            "question_type": q["question_type"],
            "difficulty": q.get("difficulty"),
            "document": q.get("document"),
        },
        "split": q["split"],
    }


def main() -> int:
    if not GOLDEN_PATH.exists():
        print("Run first: uv run python -m evals.rules.generate_golden_set", file=sys.stderr)
        return 1
    questions = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        print("Duplicate ids in the golden set", file=sys.stderr)
        return 1
    sync_dataset(DATASET_NAME, DATASET_DESCRIPTION, {q["id"]: to_example(q) for q in questions})
    return 0


if __name__ == "__main__":
    sys.exit(main())
