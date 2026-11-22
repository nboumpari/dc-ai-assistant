"""Consistency checks on the finished database.

Each check is a SQL query that counts rows breaking a rule. A healthy
database returns 0 for every check. The deliberate data errors are designed
not to break these rules, except the SSCC check digits, whose count must
match the error log exactly.
"""

import sqlite3

from .master_data import gs1_check_digit

YEAR_START, YEAR_END = "2025-01-01 00:00:00", "2025-12-31 23:59:59"

CHECKS: dict[str, str] = {
    "every pallet belongs to an existing shipment": """
        SELECT COUNT(*) FROM pallets p LEFT JOIN shipments s USING (shipment_id) WHERE s.shipment_id IS NULL
    """,
    "order line shipped qty = cases on its pallets": """
        SELECT COUNT(*) FROM order_lines ol
        LEFT JOIN (SELECT order_line_id, SUM(qty_cases) q FROM pallet_contents GROUP BY 1) pc USING (order_line_id)
        WHERE ol.qty_shipped_cases <> COALESCE(pc.q, 0)
    """,
    "transfer line shipped qty = cases on its pallets": """
        SELECT COUNT(*) FROM stock_transfer_lines tl
        LEFT JOIN (SELECT transfer_line_id, SUM(qty_cases) q FROM pallet_contents GROUP BY 1) pc
            USING (transfer_line_id)
        WHERE tl.qty_shipped_cases <> COALESCE(pc.q, 0)
    """,
    "never ship more than ordered or requested": """
        SELECT (SELECT COUNT(*) FROM order_lines WHERE qty_shipped_cases > qty_ordered_cases)
             + (SELECT COUNT(*) FROM stock_transfer_lines WHERE qty_shipped_cases > qty_requested_cases)
    """,
    "transfers never receive more than shipped": """
        SELECT COUNT(*) FROM stock_transfer_lines WHERE qty_received_cases > qty_shipped_cases
    """,
    "stock is never negative and allocation fits in stock": """
        SELECT COUNT(*) FROM inventory WHERE qty_on_hand_cases < 0 OR qty_allocated_cases > qty_on_hand_cases
    """,
    "a pallet is never received before it was loaded": """
        SELECT COUNT(*) FROM
            (SELECT sscc, MIN(event_at) t FROM pallet_events WHERE event_type = 'loaded' GROUP BY 1) l
        JOIN (SELECT sscc, MIN(event_at) t FROM pallet_events WHERE event_type = 'received' GROUP BY 1) r USING (sscc)
        WHERE r.t < l.t
    """,
    "a shipment never arrives before it leaves": """
        SELECT COUNT(*) FROM shipments WHERE actual_delivery_at < actual_ship_at
    """,
    "the plant never ships to customers": """
        SELECT COUNT(*) FROM shipments s JOIN warehouses w ON w.warehouse_id = s.origin_warehouse_id
        WHERE s.shipment_type = 'customer delivery' AND w.site_type = 'plant'
    """,
    "customer deliveries leave from the customer's home DC": """
        SELECT COUNT(*) FROM shipments s JOIN orders o USING (order_id) JOIN customers c USING (customer_id)
        WHERE s.origin_warehouse_id <> c.home_warehouse_id
    """,
    "every batch on a pallet or return is of the same product": """
        SELECT (SELECT COUNT(*) FROM pallet_contents pc JOIN production_batches b USING (batch_lot)
                WHERE pc.sku <> b.sku)
             + (SELECT COUNT(*) FROM customer_return_lines rl JOIN production_batches b USING (batch_lot)
                WHERE rl.sku <> b.sku)
    """,
    "goods are never shipped before the batch was produced": """
        SELECT COUNT(*) FROM pallet_contents pc JOIN production_batches b USING (batch_lot) JOIN pallets p USING (sscc)
        WHERE p.built_at < b.produced_at
    """,
    "operational events all happen in 2025": f"""
        SELECT (SELECT COUNT(*) FROM orders WHERE order_date NOT BETWEEN '{YEAR_START}' AND '{YEAR_END}')
             + (SELECT COUNT(*) FROM pallet_events WHERE event_at NOT BETWEEN '{YEAR_START}' AND '{YEAR_END}')
             + (SELECT COUNT(*) FROM shipment_events WHERE event_at NOT BETWEEN '{YEAR_START}' AND '{YEAR_END}')
             + (SELECT COUNT(*) FROM shipments WHERE actual_delivery_at > '{YEAR_END}')
             + (SELECT COUNT(*) FROM dock_appointments WHERE dock_end_at > '{YEAR_END}')
    """,
}


def validate(con: sqlite3.Connection, expected_bad_ssccs: int) -> list[str]:
    """Run every check. Returns a list of failures (empty means the database is consistent)."""
    failures = []

    fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
    if fk_errors:
        failures.append(f"foreign keys: {len(fk_errors)} broken references")

    for name, sql in CHECKS.items():
        broken = con.execute(sql).fetchone()[0]
        if broken:
            failures.append(f"{name}: {broken} rows break the rule")

    ssccs = [r[0] for r in con.execute("SELECT sscc FROM pallets")]
    malformed = sum(1 for s in ssccs if len(s) != 18 or not s.isdigit())
    if malformed:
        failures.append(f"SSCCs must be 18 digits: {malformed} are not")
    bad_check_digit = sum(1 for s in ssccs if gs1_check_digit(s[:17]) != s[17])
    if bad_check_digit != expected_bad_ssccs:
        failures.append(f"SSCC check digits: {bad_check_digit} wrong, but {expected_bad_ssccs} were injected")

    return failures
