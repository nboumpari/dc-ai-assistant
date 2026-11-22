"""The golden set file must be well formed and stratified. No database or network needed."""

from collections import Counter

import yaml

from evals.build_golden_set import GOLDEN_SET_PATH, check, tables_used
from evals.upload_golden_set import to_example

QUESTIONS = yaml.safe_load(GOLDEN_SET_PATH.read_text(encoding="utf-8"))["questions"]


def test_every_question_is_well_formed():
    problems = [p for q in QUESTIONS for p in check(q)]
    assert problems == []


def test_ids_are_unique():
    duplicates = [qid for qid, n in Counter(q["id"] for q in QUESTIONS).items() if n > 1]
    assert duplicates == []


def test_split_is_about_70_30():
    test_share = sum(q["split"] == "test" for q in QUESTIONS) / len(QUESTIONS)
    assert 0.25 <= test_share <= 0.35


def test_every_stratum_appears_in_both_splits():
    for stratum in {q["stratum"] for q in QUESTIONS}:
        splits = {q["split"] for q in QUESTIONS if q["stratum"] == stratum}
        assert splits == {"dev", "test"}, f"{stratum} is only in {splits}"


def test_tables_used_ignores_ctes():
    sql = "WITH refused AS (SELECT * FROM customer_returns) SELECT * FROM orders JOIN refused USING (x)"
    assert tables_used(sql) == ["customer_returns", "orders"]


def test_langsmith_example_shape():
    item = {
        "id": "L01",
        "question": "Which DC serves customer CUS012?",
        "stratum": "lookup",
        "difficulty": "easy",
        "split": "dev",
        "route": "data_agent",
        "behavior": "answer",
        "compare": "scalar",
        "key_facts": [],
        "reference_sql": "SELECT 1",
        "tables": ["customers"],
        "expected": {"columns": ["home_warehouse_id"], "rows": [["ATH"]]},
    }
    example = to_example(item)
    assert example["inputs"] == {"question": "Which DC serves customer CUS012?"}
    assert example["outputs"]["expected_rows"] == [["ATH"]]
    assert example["metadata"]["golden_id"] == "L01"
    assert example["split"] == "dev"
