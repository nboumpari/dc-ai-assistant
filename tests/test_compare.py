"""Result comparison used by the evaluation."""

import pytest

from evals.compare import results_match, row_contains, values_match


@pytest.mark.parametrize(
    "expected, actual",
    [
        (77.5, 77.5),
        (77.5, 77.49),
        (77.5, "77.5%"),
        (77.5, 0.775),  # fraction vs percent
        (0.06, 6.0),
        (7868, 7868.0),
        (1719, "1,719"),
        ("ATH", "ath"),
        ("Hermes Freight", " Hermes Freight "),
        ("2025-08", "08"),
        ("2025-08", 8),
        ("2025-08", "August"),
        ("2025-08", "aug"),
        ("August", "2025-08"),
        (None, None),
    ],
)
def test_values_that_match(expected, actual):
    assert values_match(expected, actual)


@pytest.mark.parametrize(
    "expected, actual",
    [
        (77.5, 80.0),
        (139, 89),
        ("ATH", "THE"),
        ("2025-08", "2025-07"),
        ("2025-08", 7),
        ("Monday", "Tuesday"),
        (None, 0),
        (0, None),
    ],
)
def test_values_that_do_not_match(expected, actual):
    assert not values_match(expected, actual)


def test_row_contains_ignores_column_order_and_extra_columns():
    assert row_contains(["ATH", "Athens DC", 86.8, 1234], ["ATH", 86.8])
    assert row_contains([86.8, "ATH"], ["ATH", 86.8])
    assert not row_contains(["ATH"], ["ATH", 86.8])


def test_row_contains_uses_each_value_once():
    assert not row_contains(["2025-08"], ["2025-08", "2025-08"])
    assert row_contains(["2025-08", "August"], ["2025-08", "2025-08"])


def test_scalar():
    assert results_match("scalar", [[77.5]], [[77.5]])[0]
    assert results_match("scalar", [[77.5]], [["OTIF", 77.5]])[0]
    assert not results_match("scalar", [[77.5]], [[70.0]])[0]


def test_top1_only_checks_the_first_row():
    expected = [["Ampeli Markets", 139], ["Ourania Supermarkets", 19]]
    assert results_match("top1", expected, [["CUS001", "Ampeli Markets", 139]])[0]
    assert not results_match("top1", expected, [["Ampeli Markets", 89]])[0]


def test_rows_in_any_order():
    expected = [["PAT", "Achaia Warehousing"], ["THE", "Vardar Logistics"]]
    assert results_match("rows", expected, [["THE", "Vardar Logistics"], ["PAT", "Achaia Warehousing"]])[0]
    assert not results_match("rows", expected, [["THE", "Vardar Logistics"]])[0]


def test_ordered_rows_need_the_same_order():
    expected = [["HOM-009", 883530], ["HOM-001", 534907]]
    assert results_match("ordered_rows", expected, [["HOM-009", 883530], ["HOM-001", 534907]])[0]
    assert not results_match("ordered_rows", expected, [["HOM-001", 534907], ["HOM-009", 883530]])[0]


def test_empty_result_fails_with_a_reason():
    ok, reason = results_match("scalar", [[1]], [])
    assert not ok and "no rows" in reason


ALIASES = {"car03": "car03", "hermes freight": "car03", "ath": "ath", "athens dc": "ath", "athens": "ath"}


def test_ids_and_names_match_with_aliases():
    assert values_match("Hermes Freight", "CAR03", ALIASES)
    assert values_match("ATH", "Athens", ALIASES)
    assert not values_match("Hermes Freight", "CAR01", ALIASES)
    assert not values_match("Hermes Freight", "CAR03")  # no aliases given


def test_weekday_names_match_sqlite_numbers():
    assert values_match("Monday", "1")
    assert values_match("Monday", 1)
    assert values_match("Sunday", 0)
    assert not values_match("Monday", 2)


def test_p01_style_result_passes_with_aliases():
    assert results_match("top1", [["Hermes Freight", "Monday"]], [["CAR03", "1", 221, 125, 56.6]], ALIASES)[0]


def test_values_mode_accepts_any_shape():
    expected = [["2025-08", "2025-08"]]
    actual = [["fewest_orders", "2025-08", 494, None], ["highest_no_show_rate", "2025-08", None, 4.12]]
    assert results_match("values", expected, actual)[0]
    assert not results_match("values", expected, [["fewest_orders", "2025-08"]])[0]


def test_week_start_date_matches_sqlite_week_number():
    assert values_match("2025-08-04", "31")  # strftime('%W') for the week of Monday 4 August 2025
    assert values_match("2025-08-04", 31)
    assert not values_match("2025-08-04", 32)
    assert values_match("2025-08-04", "2025-08-04")
    assert not values_match("2025-08-05", 31)  # only Monday dates stand for a week
