"""Settings for the warehouse data generator.

Everything the generator can be tuned with lives here. The output is always
one full calendar year (2025), and the same SEED always gives the same data.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

SEED = 42
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"
DB_PATH = OUTPUT_DIR / "warehouse.db"
ERRORS_LOG_PATH = OUTPUT_DIR / "injected_errors.csv"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# GS1 company prefix used to build SSCCs. Prefixes starting with "2" are GS1's
# restricted-circulation range: never assigned to a real company.
GS1_COMPANY_PREFIX = "2000000"
SSCC_EXTENSION_DIGIT = "3"


# ---------------------------------------------------------------------------
# Sites: one plant and three distribution centres
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Warehouse:
    warehouse_id: str
    name: str
    city: str
    site_type: str  # "plant" or "DC"
    operator_type: str  # "in-house" or "3PL"
    operator_name: str | None
    dock_doors: int


PLANT_ID = "PLT"

WAREHOUSES = [
    Warehouse(PLANT_ID, "Oinofyta Plant", "Oinofyta", "plant", "in-house", None, 10),
    Warehouse("ATH", "Athens DC", "Athens", "DC", "in-house", None, 12),
    Warehouse("THE", "Thessaloniki DC", "Thessaloniki", "DC", "3PL", "Vardar Logistics", 8),
    Warehouse("PAT", "Patras DC", "Patras", "DC", "3PL", "Achaia Warehousing", 6),
]

DC_IDS = [w.warehouse_id for w in WAREHOUSES if w.site_type == "DC"]

# Road transit time between sites, in hours (one way).
TRANSIT_HOURS = {
    frozenset({PLANT_ID, "ATH"}): 1.0,
    frozenset({PLANT_ID, "THE"}): 5.5,
    frozenset({PLANT_ID, "PAT"}): 3.5,
    frozenset({"ATH", "THE"}): 6.5,
    frozenset({"ATH", "PAT"}): 3.0,
    frozenset({"THE", "PAT"}): 6.0,
}

# The plant makes every product. Each production run is one batch, recorded
# in production_batches. Stock leaves the plant only as transfers to the DCs.
PLANT_PRODUCTION_WEEKDAYS = {0, 2, 4}  # Mon, Wed, Fri: each product is made once a week on one of these days
PLANT_SHIPMENT_WEEKDAYS = {0, 3}  # Mon and Thu: scheduled transfers to every DC
PLANT_PRODUCTION_DAYS_OF_COVER = 7  # one batch = about a week of network demand

# Share of customers served by each DC (the plant has no customers).
CUSTOMER_SHARE_BY_DC = {"ATH": 0.50, "THE": 0.30, "PAT": 0.20}


# ---------------------------------------------------------------------------
# Master data sizes
# ---------------------------------------------------------------------------

PRODUCTS_PER_CATEGORY = {
    "Laundry": 30,
    "Baby care": 25,
    "Hair care": 30,
    "Oral care": 20,
    "Personal care": 25,
    "Home care": 20,
}

NUM_CUSTOMERS = 80


# ---------------------------------------------------------------------------
# Carriers and their behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Carrier:
    carrier_id: str
    carrier_name: str
    mode: str  # "FTL", "LTL" or "parcel"
    cost_per_pallet: float
    late_prob: float  # chance a delivery arrives late
    no_show_prob: float  # chance the truck never shows up at the dock
    damage_prob: float  # chance a pallet gets damaged in transit


CARRIERS = [
    Carrier("CAR01", "Aegean Haulage", "FTL", 38.0, 0.06, 0.010, 0.004),
    Carrier("CAR02", "Olympos Transport", "FTL", 35.0, 0.08, 0.015, 0.006),
    Carrier("CAR03", "Hermes Freight", "FTL", 33.0, 0.08, 0.020, 0.005),  # late on Mondays (pattern)
    Carrier("CAR04", "Ionian Express", "LTL", 42.0, 0.10, 0.020, 0.008),
    Carrier("CAR05", "Pindos Logistics", "LTL", 40.0, 0.07, 0.012, 0.005),
    Carrier("CAR06", "Kentavros Cargo", "LTL", 39.0, 0.12, 0.030, 0.010),
    Carrier("CAR07", "Delta Road", "FTL", 36.0, 0.05, 0.008, 0.003),
    Carrier("CAR08", "Meteora Parcel", "parcel", 55.0, 0.09, 0.000, 0.012),
]


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------

# Average chance that a customer places an order on a working day.
DAILY_ORDER_PROB = 0.35
ORDER_LINES_MIN, ORDER_LINES_MAX = 2, 12
# Cases per order line: best sellers are ordered in bigger quantities than slow movers.
# A line's typical size = AVG_CASES_PER_LINE x sqrt(product popularity / average popularity),
# with natural variation around it.
AVG_CASES_PER_LINE = 65
LINE_QTY_VARIATION = 0.45  # spread of a lognormal distribution
LINE_QTY_MIN_CASES = 2
LINE_QTY_MAX_TYPICAL_CASES = 140  # cap on a product's typical line size

# Demand multiplier by weekday (Mon=0 ... Sun=6). No deliveries on Sunday.
WEEKDAY_FACTOR = {0: 1.25, 1: 1.05, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.35, 6: 0.0}

# Greek public holidays in 2025: no orders, no shipments.
HOLIDAYS = {
    date(2025, 1, 1),
    date(2025, 1, 6),
    date(2025, 3, 3),  # Clean Monday
    date(2025, 3, 25),
    date(2025, 4, 18),  # Good Friday
    date(2025, 4, 21),  # Easter Monday
    date(2025, 5, 1),
    date(2025, 6, 9),  # Whit Monday
    date(2025, 8, 15),
    date(2025, 10, 28),
    date(2025, 12, 25),
    date(2025, 12, 26),
}

# Stock policy, in days of expected demand.
# Plant -> DC: on plant shipment days each DC is topped up to its target cover.
DC_TARGET_DAYS_OF_COVER = {"ATH": 10, "THE": 16, "PAT": 10}  # THE (3PL) has spare space
# Safety floor: each DC keeps at least this many typical order lines of every product,
# so slow movers do not run out when one customer places a large order.
DC_MIN_STOCK_LINES = 5
PLANT_TARGET_DAYS_OF_COVER = 14  # plant production tops plant stock up to this
# DC -> DC: a DC asks another DC for stock when it drops below this...
REBALANCE_TRIGGER_DAYS = 3
REBALANCE_SOURCE_MIN_DAYS = 12  # ...and the other DC has at least this much
REBALANCE_QTY_DAYS = 7  # amount moved

TRUCK_CAPACITY_PALLETS = 33
MAX_PALLET_BUILD_HEIGHT_CM = 200


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

LATER_RETURN_PROB_PER_ORDER = 0.02  # chance an order gets a later return
LATER_RETURN_DELAY_DAYS = (3, 21)
RETURN_REASONS = ["damaged", "wrong product", "short dated", "overstock", "quality complaint", "delivered late"]
MAX_PALLET_HEIGHT_CM_DEFAULT = 180

# Chance a pallet is refused at delivery
REFUSE_IF_DAMAGED_PROB = 0.85
REFUSE_IF_TOO_TALL_PROB = 0.05  # normal customers, pallet over their limit
REFUSE_IF_TOO_TALL_STRICT_PROB = 0.6  # the strict-height customer (pattern 3)
REFUSE_IF_VERY_LATE_PROB = 0.10  # delivered more than 2 hours after the window closes


# ---------------------------------------------------------------------------
# Hidden patterns (the "stories" the data assistant should be able to find)
# ---------------------------------------------------------------------------

# 1. One carrier is often late on Mondays.
MONDAY_LATE_CARRIER = "CAR03"
MONDAY_LATE_PROB = 0.55

# 2. Summer peak for one category; DCs build up stock beforehand.
SUMMER_PEAK_CATEGORY = "Personal care"
SUMMER_PEAK_MONTHS = {6, 7, 8}
SUMMER_PEAK_FACTOR = 1.4

# 3. One customer refuses pallets taller than its height limit far more often.
STRICT_HEIGHT_CUSTOMER_INDEX = 0  # first generated customer
STRICT_HEIGHT_LIMIT_CM = 160

# 4. A delayed transfer causes a stockout at Athens in week 32 and OTIF drops.
STOCKOUT_DC = "ATH"
STOCKOUT_CATEGORY = "Baby care"
STOCKOUT_START = date(2025, 8, 4)
STOCKOUT_END = date(2025, 8, 10)
STOCKOUT_HOLD_DAYS_BEFORE = 7  # transfers are held from this many days before the stockout week

# 5. One transfer lane loses cases in transit.
LOSSY_LANE = ("THE", "ATH")
LOSSY_LANE_LOSS_PROB = 0.03  # chance a transfer line arrives short

# 6. Long dock waits at one DC in the morning peak.
SLOW_DOCK_DC = "ATH"
SLOW_DOCK_HOURS = range(6, 10)
SLOW_DOCK_EXTRA_WAIT_MIN = (30, 120)

# 7. A bad batch gets returned for quality issues by several customers.
BAD_BATCH_CATEGORY = "Hair care"
BAD_BATCH_PRODUCED = date(2025, 5, 12)
BAD_BATCH_RETURN_PROB = 0.4

# 8. August is quieter (Greek summer holidays): fewer orders, more no-shows.
AUGUST_DEMAND_FACTOR = 0.7
AUGUST_NO_SHOW_FACTOR = 3.0


# ---------------------------------------------------------------------------
# Deliberate data errors (each one is logged to ERRORS_LOG_PATH)
# ---------------------------------------------------------------------------

ERROR_RATES = {
    "duplicate_pallet_scan": 0.01,  # share of pallet scans duplicated
    "missing_loaded_scan": 0.005,  # share of pallets with the "loaded" scan removed
    "bad_sscc_check_digit": 0.002,  # share of pallets with a wrong check digit
    "transfer_pallet_never_received": 0.003,  # share of transfer pallets with no "received" scan
    "return_qty_mismatch": 0.05,  # share of return lines where received != requested
}
