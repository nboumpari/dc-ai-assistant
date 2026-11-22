"""Tests for the warehouse data generator.

A full year takes several seconds, so most tests run the same code over the
first two weeks of January only.
"""

import hashlib
import random
import sqlite3
from datetime import date, datetime, time

import pytest

from scripts.data_generator import config, simulate
from scripts.data_generator.generate import generate
from scripts.data_generator.master_data import build_master_data, gs1_check_digit


@pytest.fixture
def two_weeks(monkeypatch):
    end = date(2025, 1, 14)
    monkeypatch.setattr(config, "END_DATE", end)
    monkeypatch.setattr(simulate, "END_TS", datetime.combine(end, time(23, 59, 59)))


@pytest.fixture
def small_db(two_weeks, tmp_path):
    db, log = tmp_path / "warehouse.db", tmp_path / "injected_errors.csv"
    counts = generate(db, log)
    con = sqlite3.connect(db)
    yield con, counts, log
    con.close()


def test_gs1_check_digit_known_values():
    # SSCC example worked out by hand, and a GTIN-13 from the generated catalogue
    assert gs1_check_digit("35201234000000001") == "9"
    assert gs1_check_digit("200000000001") == "5"


def test_master_data_is_repeatable():
    a = build_master_data(random.Random(config.SEED))
    b = build_master_data(random.Random(config.SEED))
    assert a.products == b.products
    assert a.customers == b.customers


def test_master_data_sizes():
    md = build_master_data(random.Random(config.SEED))
    assert len(md.products) == sum(config.PRODUCTS_PER_CATEGORY.values())
    assert len(md.customers) == config.NUM_CUSTOMERS
    assert len({p.sku for p in md.products}) == len(md.products)
    assert all(c.home_dc in config.DC_IDS for c in md.customers)


def test_database_has_all_tables_with_rows(small_db):
    _, counts, _ = small_db
    assert len(counts) == 18
    empty = [t for t, n in counts.items() if n == 0]
    assert empty == [], f"empty tables: {empty}"


def test_simulation_is_repeatable(two_weeks):
    def fingerprint():
        rng = random.Random(config.SEED)
        rows = simulate.simulate(build_master_data(rng), rng)
        h = hashlib.sha256()
        for table in sorted(rows):
            h.update(repr(rows[table]).encode())
        return h.hexdigest()

    assert fingerprint() == fingerprint()


def test_error_log_matches_database(small_db):
    con, _, log = small_db
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "error_type,table,record_id,detail"
    logged_bad = [ln.split(",")[2] for ln in lines[1:] if ln.startswith("bad_sscc_check_digit")]
    in_db_bad = [s for (s,) in con.execute("SELECT sscc FROM pallets") if gs1_check_digit(s[:17]) != s[17]]
    assert sorted(logged_bad) == sorted(in_db_bad)


def test_plant_only_ships_to_dcs(small_db):
    con, _, _ = small_db
    destinations = {
        row[0]
        for row in con.execute(
            "SELECT DISTINCT w.site_type FROM shipments s "
            "JOIN warehouses w ON w.warehouse_id = s.destination_warehouse_id "
            "WHERE s.origin_warehouse_id = ?",
            (config.PLANT_ID,),
        )
    }
    assert destinations == {"DC"}


def test_nothing_happens_after_end_date(small_db):
    con, _, _ = small_db
    last = con.execute("SELECT MAX(event_at) FROM pallet_events").fetchone()[0]
    assert last <= "2025-01-14 23:59:59"
