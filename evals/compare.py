"""Compare the agent's query result with the expected result.

Only the data is compared, never the wording of the answer. The comparison is
tolerant of presentation:

- column names and column order do not matter, and extra columns are fine
- numbers match within 1% (or 0.05), and 0.775 matches 77.5 (fraction vs percent)
- '2025-08', '08', 8, 'August' and 'Aug' all match the same month
- 'Monday' matches SQLite's weekday number 1 (strftime('%w'): 0 = Sunday)
- a week's Monday date ('2025-08-04') matches SQLite's week number for it (strftime('%W'): 31)
- an id matches its name when aliases are given, e.g. 'CAR03' = 'Hermes Freight'
- text is compared case-insensitively, ignoring surrounding spaces
"""

import math
import re
from datetime import date
from typing import Any

MONTHS = {
    name: i
    for i, names in enumerate(
        [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
         ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
         ("december", "dec")],
        start=1,
    )
    for name in names
}  # fmt: skip

WEEKDAYS = {
    name: i
    for i, names in enumerate(
        [("sunday", "sun"), ("monday", "mon"), ("tuesday", "tue"), ("wednesday", "wed"), ("thursday", "thu"),
         ("friday", "fri"), ("saturday", "sat")]
    )
    for name in names
}  # fmt: skip

Aliases = dict[str, str]  # lower-case text -> canonical key


def _named(value: Any, names: dict[str, int], pattern: str | None = None) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if pattern and (m := re.fullmatch(pattern, text)):
        return int(m.group(1))
    return names.get(text)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").rstrip("%").strip()
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.01, abs_tol=0.05)


def _calendar_match(expected: Any, actual: Any, names: dict[str, int], pattern: str | None) -> bool | None:
    """True/False if either side names a month or weekday, None if neither does."""
    e, a = _named(expected, names, pattern), _named(actual, names, pattern)
    if e is None and a is None:
        return None
    if e is not None and a is not None:
        return e == a
    number = _number(actual) if e is not None else _number(expected)
    return number is not None and number == (e if e is not None else a)


def _week_match(expected: Any, actual: Any) -> bool | None:
    """If the expected value is a Monday date and the actual value a plain number, compare week numbers."""
    if not (isinstance(expected, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected.strip())):
        return None
    day = date.fromisoformat(expected.strip())
    number = _number(actual)
    if day.weekday() != 0 or number is None or isinstance(actual, str) and "-" in actual:
        return None
    return number == int(day.strftime("%W"))


def values_match(expected: Any, actual: Any, aliases: Aliases | None = None) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None

    week = _week_match(expected, actual)
    if week is not None:
        return week

    for names, pattern in ((MONTHS, r"\d{4}-(\d{2})"), (WEEKDAYS, None)):
        calendar = _calendar_match(expected, actual, names, pattern)
        if calendar is not None:
            return calendar

    num_e, num_a = _number(expected), _number(actual)
    if num_e is not None and num_a is not None:
        return _close(num_e, num_a) or _close(num_e, num_a * 100) or _close(num_e * 100, num_a)

    text_e, text_a = str(expected).strip().lower(), str(actual).strip().lower()
    if text_e == text_a:
        return True
    if aliases:
        key_e, key_a = aliases.get(text_e), aliases.get(text_a)
        return key_e is not None and key_e == key_a
    return False


def row_contains(actual_row: list, expected_row: list, aliases: Aliases | None = None) -> bool:
    """Every expected value matches a different value in the actual row."""
    unused = list(actual_row)
    for expected in expected_row:
        for i, actual in enumerate(unused):
            if values_match(expected, actual, aliases):
                del unused[i]
                break
        else:
            return False
    return True


def results_match(
    compare: str, expected_rows: list[list], actual_rows: list[list], aliases: Aliases | None = None
) -> tuple[bool, str]:
    """Returns (passed, reason)."""
    if not actual_rows:
        return False, "the query returned no rows"

    if compare == "scalar":
        value = expected_rows[0][0]
        ok = any(values_match(value, v, aliases) for v in actual_rows[0])
        return ok, "" if ok else f"expected {value!r} in the first row, got {list(actual_rows[0])}"

    if compare == "top1":
        ok = row_contains(actual_rows[0], expected_rows[0], aliases)
        return ok, "" if ok else f"expected first row {expected_rows[0]}, got {list(actual_rows[0])}"

    if compare == "values":
        everything = [v for row in actual_rows for v in row]
        ok = row_contains(everything, [v for row in expected_rows for v in row], aliases)
        return ok, "" if ok else f"expected values {expected_rows} somewhere in the result"

    if len(actual_rows) != len(expected_rows):
        return False, f"expected {len(expected_rows)} rows, got {len(actual_rows)}"

    if compare == "ordered_rows":
        for i, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=True)):
            if not row_contains(actual, expected, aliases):
                return False, f"row {i + 1}: expected {expected}, got {list(actual)}"
        return True, ""

    if compare == "rows":
        unused = [list(r) for r in actual_rows]
        for expected in expected_rows:
            for i, actual in enumerate(unused):
                if row_contains(actual, expected, aliases):
                    del unused[i]
                    break
            else:
                return False, f"no row matching {expected}"
        return True, ""

    raise ValueError(f"unknown compare mode: {compare}")


def load_aliases(sql_tool) -> Aliases:
    """id <-> name pairs from the master data, so 'CAR03' and 'Hermes Freight' count as the same value."""
    queries = [
        "SELECT carrier_id, carrier_name FROM carriers",
        "SELECT customer_id, customer_name FROM customers",
        "SELECT warehouse_id, name FROM warehouses",
        "SELECT warehouse_id, city FROM warehouses",
        "SELECT sku, product_name FROM products",
    ]
    aliases: Aliases = {}
    for query in queries:
        for key, name in sql_tool.run(query).rows:
            aliases[str(key).lower()] = str(key).lower()
            aliases[str(name).lower()] = str(key).lower()
    return aliases
