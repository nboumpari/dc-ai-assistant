"""Compute the expected answers of the golden set from the database.

Run from the project root:

    uv run python -m evals.build_golden_set

Reads evals/golden_set.yaml, runs every reference query through the same SQL
tool the agent uses, checks the set is well formed, and writes
evals/golden_set_expected.json (the file the LangSmith upload reads).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from dc_ai_assistant.data_assistant.sql_tool import SQLTool

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.yaml"
EXPECTED_PATH = EVALS_DIR / "golden_set_expected.json"

STRATA = {
    "lookup",
    "aggregation",
    "join",
    "time",
    "kpi",
    "pattern",
    "data_quality",
    "ambiguous",
    "refuse",
    "out_of_scope",
}
ROUTES = {"data_agent", "rules_agent", "clarify", "out_of_scope"}
BEHAVIORS = {"answer", "clarify", "refuse", "decline"}
DIFFICULTIES = {"easy", "medium", "hard"}
COMPARES = {"scalar", "rows", "ordered_rows", "top1", "values"}


def tables_used(sql: str) -> list[str]:
    names = re.findall(r"\b(?:FROM|JOIN)\s+([a-z_]+)", sql, flags=re.IGNORECASE)
    ctes = set(re.findall(r"\b([a-z_]+)\s+AS\s*\(", sql, flags=re.IGNORECASE))
    return sorted({n.lower() for n in names} - {c.lower() for c in ctes})


def check(question: dict) -> list[str]:
    qid = question.get("id", "?")
    problems = []
    for field, allowed in [
        ("stratum", STRATA),
        ("route", ROUTES),
        ("behavior", BEHAVIORS),
        ("difficulty", DIFFICULTIES),
        ("split", {"dev", "test"}),
    ]:
        if question.get(field) not in allowed:
            problems.append(f"{qid}: {field} must be one of {sorted(allowed)}")
    if question.get("behavior") == "answer":
        if not question.get("sql"):
            problems.append(f"{qid}: questions to answer need a reference sql")
        if question.get("compare") not in COMPARES:
            problems.append(f"{qid}: compare must be one of {sorted(COMPARES)}")
    elif question.get("sql"):
        problems.append(f"{qid}: only questions to answer have a reference sql")
    elif not question.get("key_facts"):
        problems.append(f"{qid}: questions without sql need key_facts describing the expected behaviour")
    return problems


def build() -> list[dict]:
    questions = yaml.safe_load(GOLDEN_SET_PATH.read_text(encoding="utf-8"))["questions"]

    problems = [p for q in questions for p in check(q)]
    duplicates = [qid for qid, n in Counter(q["id"] for q in questions).items() if n > 1]
    problems += [f"duplicate id {qid}" for qid in duplicates]

    tool = SQLTool(max_rows=1000)
    items = []
    for q in questions:
        item = {
            "id": q["id"],
            "question": q["question"],
            "stratum": q["stratum"],
            "difficulty": q["difficulty"],
            "split": q["split"],
            "route": q["route"],
            "behavior": q["behavior"],
            "compare": q.get("compare"),
            "key_facts": q.get("key_facts", []),
            "reference_sql": (q.get("sql") or "").strip() or None,
            "tables": tables_used(q.get("sql") or ""),
            "expected": None,
        }
        if item["reference_sql"]:
            result = tool.run(item["reference_sql"])
            if not result.ok:
                problems.append(f"{q['id']}: reference sql failed: {result.error}")
            elif not result.rows:
                problems.append(f"{q['id']}: reference sql returned no rows")
            elif result.truncated:
                problems.append(f"{q['id']}: reference sql returned more than 1000 rows")
            else:
                item["expected"] = {"columns": result.columns, "rows": [list(r) for r in result.rows]}
        items.append(item)

    if problems:
        raise ValueError("golden set has problems:\n  " + "\n  ".join(problems))
    return items


def main() -> int:
    try:
        items = build()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    EXPECTED_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    splits = Counter(i["split"] for i in items)
    print(f"{len(items)} questions: {splits['dev']} dev, {splits['test']} test\n")
    print(f"{'stratum':<14}{'dev':>5}{'test':>6}")
    for stratum in sorted(STRATA):
        dev = sum(1 for i in items if i["stratum"] == stratum and i["split"] == "dev")
        test = sum(1 for i in items if i["stratum"] == stratum and i["split"] == "test")
        print(f"{stratum:<14}{dev:>5}{test:>6}")
    print(f"\nDifficulty: {dict(Counter(i['difficulty'] for i in items))}")
    print(f"Routes:     {dict(Counter(i['route'] for i in items))}")
    print(f"\nWrote {EXPECTED_PATH.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
