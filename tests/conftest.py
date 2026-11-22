"""Shared fixtures: a small generated database (first two weeks of 2025), built once per test run."""

import os

# Tests never send traces to LangSmith (set before anything loads .env).
os.environ["LANGSMITH_TRACING"] = "false"

import sqlite3  # noqa: E402
from datetime import date, datetime, time  # noqa: E402

import pytest  # noqa: E402

from scripts.data_generator import config, simulate  # noqa: E402
from scripts.data_generator.generate import generate  # noqa: E402

TEST_END = date(2025, 1, 14)


def _limit_to_two_weeks(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(config, "END_DATE", TEST_END)
    mp.setattr(simulate, "END_TS", datetime.combine(TEST_END, time(23, 59, 59)))


@pytest.fixture
def two_weeks(monkeypatch):
    _limit_to_two_weeks(monkeypatch)


@pytest.fixture(scope="session")
def small_db_files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("generated")
    db, log = folder / "warehouse.db", folder / "injected_errors.csv"
    with pytest.MonkeyPatch.context() as mp:
        _limit_to_two_weeks(mp)
        counts = generate(db, log)
    return db, log, counts


@pytest.fixture(scope="session")
def small_db(small_db_files):
    db, log, counts = small_db_files
    con = sqlite3.connect(db)
    yield con, counts, log
    con.close()


@pytest.fixture
def empty_db():
    """The schema with no data, for checks that only need table and column names."""
    con = sqlite3.connect(":memory:")
    con.executescript(config.SCHEMA_PATH.read_text(encoding="utf-8"))
    yield con
    con.close()
