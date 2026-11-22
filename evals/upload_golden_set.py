"""Upload (or sync) the golden set to a LangSmith dataset.

Run from the project root, after build_golden_set:

    uv run python -m evals.upload_golden_set

Needs LANGSMITH_API_KEY in .env. Safe to run again: examples are matched by
their golden-set id, so changed questions are updated, new ones added and
removed ones deleted. The dataset keeps its history and past experiments.

In LangSmith each example has:
  inputs    {question}
  outputs   expected route, behaviour, result, key facts and reference SQL
  metadata  golden_id, stratum, difficulty, tables (filter and group by these)
  split     dev or test
"""

import json
import sys

from langsmith import Client

from dc_ai_assistant import settings  # noqa: F401  (loads .env)
from evals.build_golden_set import EXPECTED_PATH

DATASET_NAME = "dc-data-agent-golden-set"
DATASET_DESCRIPTION = (
    "Stratified golden set for the warehouse data agent and the supervisor's routing. "
    "Expected results are computed from the 2025 warehouse database. "
    "Splits: dev (prompt tuning) and test (final score)."
)


def to_example(item: dict) -> dict:
    expected = item.get("expected") or {}
    return {
        "inputs": {"question": item["question"]},
        "outputs": {
            "route": item["route"],
            "behavior": item["behavior"],
            "compare": item["compare"],
            "expected_columns": expected.get("columns"),
            "expected_rows": expected.get("rows"),
            "key_facts": item["key_facts"],
            "reference_sql": item["reference_sql"],
        },
        "metadata": {
            "golden_id": item["id"],
            "stratum": item["stratum"],
            "difficulty": item["difficulty"],
            "tables": item["tables"],
        },
        "split": item["split"],
    }


def _current_split(example) -> str | None:
    split = (example.metadata or {}).get("dataset_split")
    return split[0] if isinstance(split, list) and split else split


def _unchanged(existing, wanted: dict) -> bool:
    metadata = {k: v for k, v in (existing.metadata or {}).items() if k != "dataset_split"}
    return (
        existing.inputs == wanted["inputs"]
        and existing.outputs == wanted["outputs"]
        and metadata == wanted["metadata"]
        and _current_split(existing) == wanted["split"]
    )


def main() -> int:
    if not EXPECTED_PATH.exists():
        print("Run first: uv run python -m evals.build_golden_set", file=sys.stderr)
        return 1
    wanted = {item["id"]: to_example(item) for item in json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))}

    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(DATASET_NAME, description=DATASET_DESCRIPTION)
        print(f"Created dataset '{DATASET_NAME}'")

    existing = {(ex.metadata or {}).get("golden_id"): ex for ex in client.list_examples(dataset_id=dataset.id)}

    created = updated = unchanged = deleted = 0
    for golden_id, example in wanted.items():
        current = existing.get(golden_id)
        if current is None:
            client.create_example(dataset_id=dataset.id, **example)
            created += 1
        elif _unchanged(current, example):
            unchanged += 1
        else:
            client.update_example(current.id, dataset_id=dataset.id, **example)
            updated += 1
    for golden_id, ex in existing.items():
        if golden_id not in wanted:
            client.delete_example(ex.id)
            deleted += 1

    print(f"Dataset '{DATASET_NAME}' ({dataset.id})")
    print(f"  created {created}, updated {updated}, unchanged {unchanged}, deleted {deleted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
