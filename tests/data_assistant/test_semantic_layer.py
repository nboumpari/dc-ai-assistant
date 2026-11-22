"""The semantic layer must match the real database, so it can never drift."""

import re

import pytest

from dc_ai_assistant.data_assistant.semantic_layer import (
    load_semantic_layer,
    render_semantic_layer,
    sensitive_columns,
)

LAYER = load_semantic_layer()


def db_columns(con) -> dict[str, set[str]]:
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")} for t in tables}


def test_every_table_is_described(empty_db):
    assert set(LAYER["tables"]) == set(db_columns(empty_db))


def test_every_column_is_described(empty_db):
    actual = db_columns(empty_db)
    for table, spec in LAYER["tables"].items():
        described = set(spec["columns"])
        assert described == actual[table], (
            f"{table}: missing {actual[table] - described}, unknown {described - actual[table]}"
        )


def test_every_column_has_a_description_or_values():
    for table, spec in LAYER["tables"].items():
        for column, col in spec["columns"].items():
            assert col.get("description") or col.get("values"), f"{table}.{column} is undocumented"


def test_relationships_use_real_columns(empty_db):
    actual = db_columns(empty_db)
    for rel in LAYER["relationships"]:
        for side in rel.split(" = "):
            table, column = side.strip().split(".")
            assert column in actual[table], f"bad relationship: {rel}"


def test_kpi_split_by_uses_real_columns(empty_db):
    actual = db_columns(empty_db)
    for key, kpi in LAYER["kpis"].items():
        for expr in kpi["split_by"].values():
            for table, column in re.findall(r"\b([a-z_]+)\.([a-z_]+)\b", expr):
                assert column in actual.get(table, set()), f"{key}: unknown column {table}.{column}"


def test_listed_values_cover_the_data(small_db):
    con, _, _ = small_db
    for table, spec in LAYER["tables"].items():
        for column, col in spec["columns"].items():
            if "values" not in col:
                continue
            in_data = {r[0] for r in con.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL")}
            unknown = in_data - set(col["values"])
            assert not unknown, f"{table}.{column} has values not in the semantic layer: {unknown}"


@pytest.mark.parametrize("key", list(LAYER["kpis"]))
def test_kpi_sql_runs(small_db, key):
    con, _, _ = small_db
    kpi = LAYER["kpis"][key]
    rows = con.execute(kpi["sql"]).fetchall()
    assert len(rows) == 1 and len(rows[0]) == 1
    value = rows[0][0]
    if value is not None and kpi["unit"] == "percent":
        assert 0 <= value <= 100


def test_sensitive_columns():
    assert sensitive_columns() == {("carriers", "cost_per_pallet")}


def test_rendered_text_mentions_every_table_and_kpi():
    text = render_semantic_layer()
    for table in LAYER["tables"]:
        assert f"### {table}" in text
    for key in LAYER["kpis"]:
        assert f"({key})" in text
    assert "SENSITIVE" in text
