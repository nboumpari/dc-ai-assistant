"""Generate the warehouse database for 2025.

Run from the project root:

    uv run python -m scripts.data_generator.generate

Output:
    data/generated/warehouse.db          the SQLite database (18 tables)
    data/generated/injected_errors.csv   every deliberate data error
"""

import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from . import config
from .errors import inject_errors, write_log
from .master_data import build_master_data, insert_master_data
from .simulate import simulate
from .validate import validate

# parents before children, so foreign keys always point to existing rows
INSERT_ORDER = [
    "production_batches",
    "orders",
    "order_lines",
    "stock_transfers",
    "stock_transfer_lines",
    "customer_returns",
    "customer_return_lines",
    "shipments",
    "shipment_events",
    "pallets",
    "pallet_contents",
    "pallet_events",
    "inventory",
    "dock_appointments",
]


def generate(db_path: Path = config.DB_PATH, errors_path: Path = config.ERRORS_LOG_PATH) -> dict[str, int]:
    """Build the database from scratch. Returns the row count per table."""
    rng = random.Random(config.SEED)
    master = build_master_data(rng)
    rows = simulate(master, rng)
    log = inject_errors(rows, rng)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)

    con = sqlite3.connect(tmp_path)
    try:
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("PRAGMA synchronous = OFF")
        con.executescript(config.SCHEMA_PATH.read_text(encoding="utf-8"))
        with con:
            insert_master_data(con, master)
            for table in INSERT_ORDER:
                if rows[table]:
                    placeholders = ", ".join("?" * len(rows[table][0]))
                    con.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows[table])

        expected_bad_ssccs = sum(1 for e in log if e[0] == "bad_sscc_check_digit")
        failures = validate(con, expected_bad_ssccs)
        if failures:
            raise RuntimeError("validation failed:\n  " + "\n  ".join(failures))

        counts = {
            name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        }
    finally:
        con.close()

    # only replace the old database once the new one is complete and valid
    tmp_path.replace(db_path)
    write_log(log, errors_path)
    return counts


def main() -> int:
    start = time.perf_counter()
    print(f"Generating {config.START_DATE} to {config.END_DATE} (seed {config.SEED})...")
    try:
        counts = generate()
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    width = max(len(t) for t in counts)
    print()
    for table, n in counts.items():
        print(f"  {table:<{width}}  {n:>10,}")
    print(f"  {'total':<{width}}  {sum(counts.values()):>10,}")

    errors = Counter(line.split(",")[0] for line in config.ERRORS_LOG_PATH.read_text(encoding="utf-8").splitlines()[1:])
    print("\nDeliberate data errors:")
    for kind, n in sorted(errors.items()):
        print(f"  {kind:<32}  {n:>6,}")

    print(f"\nAll validation checks passed in {time.perf_counter() - start:.1f}s")
    print(f"Database:   {config.DB_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Error log:  {config.ERRORS_LOG_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
