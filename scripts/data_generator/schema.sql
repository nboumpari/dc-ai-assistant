-- Warehouse database: 18 tables.
-- All data is synthetic. Timestamps are ISO 8601 text (YYYY-MM-DD HH:MM:SS).

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Master data
-- ---------------------------------------------------------------------------

CREATE TABLE products (
    sku                TEXT PRIMARY KEY,
    product_name       TEXT NOT NULL,
    brand              TEXT NOT NULL,
    category           TEXT NOT NULL,
    gtin               TEXT NOT NULL UNIQUE,
    units_per_case     INTEGER NOT NULL,
    cases_per_layer    INTEGER NOT NULL,
    cases_per_pallet   INTEGER NOT NULL,
    case_weight_kg     REAL NOT NULL,
    case_height_cm     INTEGER NOT NULL,
    shelf_life_days    INTEGER NOT NULL
);

CREATE TABLE warehouses (
    warehouse_id   TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    city           TEXT NOT NULL,
    country        TEXT NOT NULL,
    site_type      TEXT NOT NULL CHECK (site_type IN ('plant', 'DC')),
    operator_type  TEXT NOT NULL CHECK (operator_type IN ('in-house', '3PL')),
    operator_name  TEXT
);

CREATE TABLE customers (
    customer_id            TEXT PRIMARY KEY,
    customer_name          TEXT NOT NULL,
    region                 TEXT NOT NULL,
    country                TEXT NOT NULL,
    home_warehouse_id      TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    delivery_window_start  TEXT NOT NULL,  -- HH:MM
    delivery_window_end    TEXT NOT NULL,  -- HH:MM
    max_pallet_height_cm   INTEGER NOT NULL
);

CREATE TABLE carriers (
    carrier_id       TEXT PRIMARY KEY,
    carrier_name     TEXT NOT NULL,
    mode             TEXT NOT NULL CHECK (mode IN ('FTL', 'LTL', 'parcel')),
    cost_per_pallet  REAL NOT NULL
);

-- ---------------------------------------------------------------------------
-- Production at the plant
-- ---------------------------------------------------------------------------

CREATE TABLE production_batches (
    batch_lot         TEXT PRIMARY KEY,
    sku               TEXT NOT NULL REFERENCES products (sku),
    warehouse_id      TEXT NOT NULL REFERENCES warehouses (warehouse_id),  -- the plant
    produced_at       TEXT NOT NULL,
    qty_cases         INTEGER NOT NULL,
    best_before_date  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Customer orders (outbound)
-- ---------------------------------------------------------------------------

CREATE TABLE orders (
    order_id                 TEXT PRIMARY KEY,
    customer_id              TEXT NOT NULL REFERENCES customers (customer_id),
    warehouse_id             TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    order_date               TEXT NOT NULL,
    requested_delivery_date  TEXT NOT NULL,
    cutoff_time              TEXT NOT NULL,
    status                   TEXT NOT NULL
        CHECK (status IN ('open', 'picked', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE order_lines (
    order_line_id      TEXT PRIMARY KEY,
    order_id           TEXT NOT NULL REFERENCES orders (order_id),
    sku                TEXT NOT NULL REFERENCES products (sku),
    qty_ordered_cases  INTEGER NOT NULL,
    qty_shipped_cases  INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- Stock transfers: plant -> DC and DC -> DC (outbound for the sender, inbound for the receiver)
-- ---------------------------------------------------------------------------

CREATE TABLE stock_transfers (
    transfer_id             TEXT PRIMARY KEY,
    from_warehouse_id       TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    to_warehouse_id         TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    created_at              TEXT NOT NULL,
    requested_arrival_date  TEXT NOT NULL,
    reason                  TEXT NOT NULL
        CHECK (reason IN ('replenishment', 'rebalancing', 'peak build-up', 'consolidation')),
    status                  TEXT NOT NULL
        CHECK (status IN ('open', 'shipped', 'received', 'cancelled')),
    CHECK (from_warehouse_id <> to_warehouse_id)
);

CREATE TABLE stock_transfer_lines (
    transfer_line_id     TEXT PRIMARY KEY,
    transfer_id          TEXT NOT NULL REFERENCES stock_transfers (transfer_id),
    sku                  TEXT NOT NULL REFERENCES products (sku),
    qty_requested_cases  INTEGER NOT NULL,
    qty_shipped_cases    INTEGER NOT NULL DEFAULT 0,
    qty_received_cases   INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- Customer returns (inbound)
-- ---------------------------------------------------------------------------

CREATE TABLE customer_returns (
    return_id          TEXT PRIMARY KEY,
    customer_id        TEXT NOT NULL REFERENCES customers (customer_id),
    warehouse_id       TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    original_order_id  TEXT REFERENCES orders (order_id),
    return_type        TEXT NOT NULL CHECK (return_type IN ('refused at delivery', 'later return')),
    requested_at       TEXT NOT NULL,
    reason_code        TEXT NOT NULL,
    status             TEXT NOT NULL
        CHECK (status IN ('requested', 'approved', 'collected', 'received', 'inspected', 'closed', 'rejected'))
);

CREATE TABLE customer_return_lines (
    return_line_id          TEXT PRIMARY KEY,
    return_id               TEXT NOT NULL REFERENCES customer_returns (return_id),
    original_order_line_id  TEXT REFERENCES order_lines (order_line_id),
    sku                     TEXT NOT NULL REFERENCES products (sku),
    batch_lot               TEXT REFERENCES production_batches (batch_lot),
    qty_requested_cases     INTEGER NOT NULL,
    qty_received_cases      INTEGER NOT NULL DEFAULT 0,
    condition               TEXT
        CHECK (condition IN ('sellable', 'damaged packaging', 'damaged product', 'expired')),
    disposition             TEXT
        CHECK (disposition IN ('restock', 'rework', 'destroy', 'return to plant')),
    inspected_at            TEXT
);

-- ---------------------------------------------------------------------------
-- Shipments (one table for deliveries, transfers and returns)
-- ---------------------------------------------------------------------------

CREATE TABLE shipments (
    shipment_id               TEXT PRIMARY KEY,
    shipment_type             TEXT NOT NULL
        CHECK (shipment_type IN ('customer delivery', 'stock transfer', 'customer return')),
    order_id                  TEXT REFERENCES orders (order_id),
    transfer_id               TEXT REFERENCES stock_transfers (transfer_id),
    return_id                 TEXT REFERENCES customer_returns (return_id),
    carrier_id                TEXT NOT NULL REFERENCES carriers (carrier_id),
    origin_warehouse_id       TEXT REFERENCES warehouses (warehouse_id),
    origin_customer_id        TEXT REFERENCES customers (customer_id),
    destination_warehouse_id  TEXT REFERENCES warehouses (warehouse_id),
    planned_ship_at           TEXT NOT NULL,
    actual_ship_at            TEXT,
    planned_delivery_at       TEXT NOT NULL,
    actual_delivery_at        TEXT,
    status                    TEXT NOT NULL
        CHECK (status IN ('planned', 'loaded', 'in transit', 'delivered', 'failed')),
    -- exactly one source document, matching the shipment type
    CHECK (
        (shipment_type = 'customer delivery' AND order_id IS NOT NULL AND transfer_id IS NULL AND return_id IS NULL)
     OR (shipment_type = 'stock transfer'    AND transfer_id IS NOT NULL AND order_id IS NULL AND return_id IS NULL)
     OR (shipment_type = 'customer return'   AND return_id IS NOT NULL AND order_id IS NULL AND transfer_id IS NULL)
    )
);

CREATE TABLE shipment_events (
    event_id     TEXT PRIMARY KEY,
    shipment_id  TEXT NOT NULL REFERENCES shipments (shipment_id),
    event_type   TEXT NOT NULL
        CHECK (event_type IN ('departed', 'delayed', 'arrived', 'delivered', 'failed delivery')),
    event_at     TEXT NOT NULL,
    reason_code  TEXT,
    location     TEXT
);

-- ---------------------------------------------------------------------------
-- Pallets (SSCC)
-- ---------------------------------------------------------------------------

CREATE TABLE pallets (
    sscc             TEXT PRIMARY KEY,  -- 18 digits
    shipment_id      TEXT REFERENCES shipments (shipment_id),
    warehouse_id     TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    built_at         TEXT NOT NULL,
    pallet_type      TEXT NOT NULL CHECK (pallet_type IN ('single-SKU', 'mixed')),
    height_cm        INTEGER NOT NULL,
    gross_weight_kg  REAL NOT NULL
);

CREATE TABLE pallet_contents (
    sscc              TEXT NOT NULL REFERENCES pallets (sscc),
    sku               TEXT NOT NULL REFERENCES products (sku),
    order_line_id     TEXT REFERENCES order_lines (order_line_id),
    transfer_line_id  TEXT REFERENCES stock_transfer_lines (transfer_line_id),
    return_line_id    TEXT REFERENCES customer_return_lines (return_line_id),
    qty_cases         INTEGER NOT NULL,
    batch_lot         TEXT NOT NULL REFERENCES production_batches (batch_lot),
    best_before_date  TEXT NOT NULL,
    -- exactly one source line
    CHECK (
        (order_line_id IS NOT NULL) + (transfer_line_id IS NOT NULL) + (return_line_id IS NOT NULL) = 1
    )
);

CREATE TABLE pallet_events (
    event_id    TEXT PRIMARY KEY,
    sscc        TEXT NOT NULL REFERENCES pallets (sscc),
    event_type  TEXT NOT NULL
        CHECK (event_type IN ('built', 'staged', 'loaded', 'departed', 'received', 'damaged', 'rejected')),
    event_at    TEXT NOT NULL,
    location    TEXT NOT NULL,
    scanned_by  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Stock
-- ---------------------------------------------------------------------------

CREATE TABLE inventory (
    snapshot_date        TEXT NOT NULL,
    warehouse_id         TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    sku                  TEXT NOT NULL REFERENCES products (sku),
    qty_on_hand_cases    INTEGER NOT NULL,
    qty_allocated_cases  INTEGER NOT NULL,
    PRIMARY KEY (snapshot_date, warehouse_id, sku)
);

-- ---------------------------------------------------------------------------
-- Docks
-- ---------------------------------------------------------------------------

CREATE TABLE dock_appointments (
    appointment_id     TEXT PRIMARY KEY,
    warehouse_id       TEXT NOT NULL REFERENCES warehouses (warehouse_id),
    dock_door          TEXT NOT NULL,
    direction          TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    shipment_id        TEXT REFERENCES shipments (shipment_id),
    carrier_id         TEXT NOT NULL REFERENCES carriers (carrier_id),
    truck_plate        TEXT NOT NULL,
    slot_start         TEXT NOT NULL,
    slot_end           TEXT NOT NULL,
    checked_in_at      TEXT,
    dock_start_at      TEXT,
    dock_end_at        TEXT,
    status             TEXT NOT NULL
        CHECK (status IN ('booked', 'arrived', 'in progress', 'completed', 'no-show', 'cancelled', 'rescheduled')),
    delay_reason_code  TEXT
);

-- ---------------------------------------------------------------------------
-- Indexes for common lookups
-- ---------------------------------------------------------------------------

CREATE INDEX idx_production_batches_sku ON production_batches (sku, produced_at);
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date ON orders (order_date);
CREATE INDEX idx_order_lines_order ON order_lines (order_id);
CREATE INDEX idx_transfer_lines_transfer ON stock_transfer_lines (transfer_id);
CREATE INDEX idx_return_lines_return ON customer_return_lines (return_id);
CREATE INDEX idx_shipments_order ON shipments (order_id);
CREATE INDEX idx_shipments_transfer ON shipments (transfer_id);
CREATE INDEX idx_shipment_events_shipment ON shipment_events (shipment_id);
CREATE INDEX idx_pallets_shipment ON pallets (shipment_id);
CREATE INDEX idx_pallet_contents_sscc ON pallet_contents (sscc);
CREATE INDEX idx_pallet_contents_batch ON pallet_contents (batch_lot);
CREATE INDEX idx_pallet_events_sscc ON pallet_events (sscc);
CREATE INDEX idx_dock_appointments_wh_slot ON dock_appointments (warehouse_id, slot_start);
