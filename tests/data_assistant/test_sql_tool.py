"""The SQL tool must allow reads and block everything else."""

import pytest

from dc_ai_assistant.data_assistant.sql_tool import SQLTool


@pytest.fixture
def tool(small_db_files):
    return SQLTool(db_path=small_db_files[0], max_rows=5, timeout_seconds=1)


def test_simple_select(tool):
    result = tool.run("SELECT warehouse_id, site_type FROM warehouses ORDER BY warehouse_id")
    assert result.ok
    assert result.columns == ["warehouse_id", "site_type"]
    assert ("PLT", "plant") in result.rows


def test_join_and_cte(tool):
    result = tool.run(
        "WITH c AS (SELECT customer_id, home_warehouse_id FROM customers) "
        "SELECT w.name, COUNT(*) FROM c JOIN warehouses w ON w.warehouse_id = c.home_warehouse_id GROUP BY 1"
    )
    assert result.ok and len(result.rows) == 3


def test_trailing_semicolon_is_fine(tool):
    assert tool.run("SELECT COUNT(*) FROM products;").ok


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM orders",
        "UPDATE orders SET status = 'cancelled'",
        "INSERT INTO warehouses VALUES ('X', 'x', 'x', 'x', 'DC', 'in-house', NULL)",
        "DROP TABLE orders",
        "CREATE TABLE t (x INT)",
        "ALTER TABLE orders ADD COLUMN x INT",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'other.db' AS other",
    ],
)
def test_anything_but_reading_is_blocked(tool, query):
    result = tool.run(query)
    assert not result.ok
    assert "Not allowed" in result.error or "readonly" in result.error.lower()


@pytest.mark.parametrize(
    "query",
    [
        "SELECT cost_per_pallet FROM carriers",
        "SELECT * FROM carriers",
        "SELECT carrier_name FROM carriers WHERE cost_per_pallet > 30",
        "SELECT (SELECT MAX(cost_per_pallet) FROM carriers)",
        "SELECT s.shipment_id, c.cost_per_pallet FROM shipments s JOIN carriers c USING (carrier_id)",
    ],
)
def test_sensitive_column_is_blocked_everywhere(tool, query):
    result = tool.run(query)
    assert not result.ok
    assert "carriers.cost_per_pallet" in result.error


def test_other_carrier_columns_are_readable(tool):
    result = tool.run("SELECT carrier_id, carrier_name, mode FROM carriers")
    assert result.ok and len(result.rows) == 5  # truncated at max_rows
    assert result.truncated


def test_row_limit(tool):
    result = tool.run("SELECT sku FROM products")
    assert len(result.rows) == 5
    assert result.truncated
    assert "truncated" in result.to_text()


def test_timeout_stops_long_queries(tool):
    result = tool.run("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n")
    assert not result.ok
    assert "stopped after" in result.error


def test_only_one_statement(tool):
    result = tool.run("SELECT 1; SELECT 2")
    assert not result.ok


def test_syntax_error_is_returned_not_raised(tool):
    result = tool.run("SELEC sku FROM products")
    assert not result.ok
    assert "syntax error" in result.error


def test_unknown_column_message_helps_the_agent(tool):
    result = tool.run("SELECT colour FROM products")
    assert not result.ok
    assert "no such column" in result.error


def test_text_output(tool):
    text = tool.run("SELECT warehouse_id, operator_name FROM warehouses WHERE warehouse_id = 'PLT'").to_text()
    assert text.splitlines()[0] == "warehouse_id | operator_name"
    assert "PLT | NULL" in text
    assert text.endswith("(1 rows)")


def test_missing_database_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_generator.generate"):
        SQLTool(db_path=tmp_path / "nope.db")
