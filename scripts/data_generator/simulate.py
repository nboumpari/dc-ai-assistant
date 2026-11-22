"""Day-by-day simulation of the plant, the three DCs and their customers for 2025.

Every row in the operational tables comes from a simulated event, so the
tables agree with each other: shipped cases equal the cases on pallets,
stock goes down when goods leave and up when they arrive, and every case
can be traced back to the production batch it came from.

Each shipment's whole life (dock, loading, transit, delivery, refusals) is
worked out when it is dispatched. Events that would happen after
31 Dec 2025 are not written, so the year ends with some trucks in transit.
"""

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from . import config
from .master_data import Customer, MasterData, Product, gs1_check_digit

END_TS = datetime.combine(config.END_DATE, time(23, 59, 59))
ISLAND_REGIONS = {"Cyclades", "Dodecanese", "Crete", "North Aegean", "Ionian Islands"}
PALLET_BASE_CM = 15
PALLET_TARE_KG = 25.0


def ts(dt: datetime | None) -> str | None:
    return None if dt is None else dt.strftime("%Y-%m-%d %H:%M:%S")


def at(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(d, time(hour, minute))


def is_operating_day(d: date) -> bool:
    return d.weekday() != 6 and d not in config.HOLIDAYS


def next_operating_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_operating_day(d):
        d += timedelta(days=1)
    return d


def happened(dt: datetime | None) -> bool:
    return dt is not None and dt <= END_TS


# ---------------------------------------------------------------------------
# Stock held per site and product, as batches in first-in-first-out order
# ---------------------------------------------------------------------------


class Stock:
    def __init__(self) -> None:
        self._batches: dict[tuple[str, str], deque[list]] = defaultdict(deque)
        self._total: dict[tuple[str, str], int] = defaultdict(int)

    def total(self, site: str, sku: str) -> int:
        return self._total[(site, sku)]

    def add(self, site: str, sku: str, lot: str, qty: int) -> None:
        if qty > 0:
            self._batches[(site, sku)].append([lot, qty])
            self._total[(site, sku)] += qty

    def take(self, site: str, sku: str, qty: int) -> list[tuple[str, int]]:
        """Take up to qty cases, oldest batch first. Returns [(lot, qty), ...]."""
        taken = []
        queue = self._batches[(site, sku)]
        while qty > 0 and queue:
            lot, available = queue[0]
            n = min(qty, available)
            taken.append((lot, n))
            qty -= n
            self._total[(site, sku)] -= n
            if n == available:
                queue.popleft()
            else:
                queue[0][1] -= n
        return taken


# ---------------------------------------------------------------------------
# Small working records
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """Cases of one batch going onto a pallet, linked to their source line."""

    sku: str
    lot: str
    qty: int
    order_line_id: str | None = None
    transfer_line_id: str | None = None
    return_line_id: str | None = None


@dataclass
class Pallet:
    sscc: str
    items: list[Item]
    height_cm: int
    weight_kg: float


@dataclass
class OrderLine:
    order_line_id: str
    sku: str
    qty_ordered: int
    qty_shipped: int = 0
    lots: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class Order:
    order_id: str
    customer: Customer
    order_date: datetime
    ship_day: date
    requested_delivery: date
    cutoff: datetime
    lines: list[OrderLine]
    status: str = "open"


@dataclass
class TransferLine:
    transfer_line_id: str
    sku: str
    qty_requested: int
    qty_shipped: int = 0
    qty_received: int = 0


@dataclass
class Transfer:
    transfer_id: str
    source: str
    dest: str
    created_at: datetime
    ship_day: date
    requested_arrival: date
    reason: str
    lines: list[TransferLine]
    status: str = "open"


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------


class Simulation:
    def __init__(self, md: MasterData, rng: random.Random) -> None:
        self.md = md
        self.rng = rng
        self.products = {p.sku: p for p in md.products}
        self.carriers = {c.carrier_id: c for c in md.carriers}
        self.sites = {w.warehouse_id: w for w in md.warehouses}
        self.stock = Stock()

        self.rows: dict[str, list[tuple]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._sscc_serial = 0

        self.orders: list[Order] = []
        self.transfers: list[Transfer] = []
        self.pending_orders: list[Order] = []  # placed, ship on their ship_day
        self.pending_transfers: list[Transfer] = []  # created, ship on their ship_day
        self.arrivals: dict[date, list[tuple[str, str, str, int]]] = defaultdict(list)  # put-away date -> stock
        self.in_transit: dict[tuple[str, str], int] = defaultdict(int)  # (dest, sku) -> cases on the way
        self.scheduled_returns: dict[date, list[tuple[Order, list[OrderLine], str]]] = defaultdict(list)

        # Fixed per-customer behaviour
        self.customer_transit_h: dict[str, float] = {}
        self.preferred_carrier: dict[tuple[str, str], str] = {}
        ltl = [c.carrier_id for c in md.carriers if c.mode == "LTL"]
        ftl = [c.carrier_id for c in md.carriers if c.mode == "FTL"]
        for c in md.customers:
            island = c.region in ISLAND_REGIONS
            self.customer_transit_h[c.customer_id] = rng.uniform(9, 14) if island else rng.uniform(1, 5)
            self.preferred_carrier[(c.customer_id, "LTL")] = rng.choice(ltl)
            self.preferred_carrier[(c.customer_id, "FTL")] = rng.choice(ftl)
        self.ltl_carriers, self.ftl_carriers = ltl, ftl
        self.parcel_carriers = [c.carrier_id for c in md.carriers if c.mode == "parcel"]

        # Production day of the week for each product (Mon, Wed or Fri)
        weekdays = sorted(config.PLANT_PRODUCTION_WEEKDAYS)
        self.production_weekday = {p.sku: weekdays[i % len(weekdays)] for i, p in enumerate(md.products)}

        # Typical order line size per product: best sellers are ordered in bigger quantities
        mean_pop = sum(p.popularity for p in md.products) / len(md.products)
        self.typical_line_cases = {
            p.sku: min(
                config.LINE_QTY_MAX_TYPICAL_CASES,
                max(config.LINE_QTY_MIN_CASES, config.AVG_CASES_PER_LINE * (p.popularity / mean_pop) ** 0.5),
            )
            for p in md.products
        }

        # Expected daily demand per DC and product (before seasonality)
        self.base_demand = self._expected_demand()

        # Pattern 7: the bad batch is the most popular hair care product's batch made in that week
        hair = [p for p in md.products if p.category == config.BAD_BATCH_CATEGORY]
        self.bad_batch_sku = max(hair, key=lambda p: p.popularity).sku
        self.bad_batch_lot: str | None = None

        self.popularity_skus = [p.sku for p in md.products]

    # -- ids --------------------------------------------------------------

    def new_id(self, prefix: str, width: int) -> str:
        self._counters[prefix] += 1
        return f"{prefix}-{self._counters[prefix]:0{width}d}"

    def new_sscc(self) -> str:
        self._sscc_serial += 1
        body = f"{config.SSCC_EXTENSION_DIGIT}{config.GS1_COMPANY_PREFIX}{self._sscc_serial:09d}"
        return body + gs1_check_digit(body)

    def truck_plate(self) -> str:
        letters = "ABEHIKMNOPTXYZ"
        return "".join(self.rng.choice(letters) for _ in range(3)) + f"-{self.rng.randint(1000, 9999)}"

    # -- demand -----------------------------------------------------------

    def _expected_demand(self) -> dict[str, dict[str, float]]:
        total_pop = sum(p.popularity for p in self.md.products)
        avg_lines = (config.ORDER_LINES_MIN + config.ORDER_LINES_MAX) / 2
        operating = [f for f in config.WEEKDAY_FACTOR.values() if f > 0]
        avg_weekday = sum(operating) / len(operating)
        demand: dict[str, dict[str, float]] = {dc: defaultdict(float) for dc in config.DC_IDS}
        for c in self.md.customers:
            orders_per_day = config.DAILY_ORDER_PROB * c.order_frequency * avg_weekday
            for p in self.md.products:
                line_share = p.popularity / total_pop
                demand[c.home_dc][p.sku] += orders_per_day * avg_lines * line_share * self.typical_line_cases[p.sku]
        return demand

    def line_quantity(self, sku: str) -> int:
        sigma = config.LINE_QTY_VARIATION
        qty = self.typical_line_cases[sku] * self.rng.lognormvariate(-(sigma**2) / 2, sigma)
        return max(config.LINE_QTY_MIN_CASES, round(qty))

    def dc_target(self, dc: str, sku: str, d: date) -> float:
        """Stock a DC aims to hold: days of cover, but never below the safety floor."""
        cover = self.daily_demand(dc, sku, d) * config.DC_TARGET_DAYS_OF_COVER[dc]
        return max(cover, config.DC_MIN_STOCK_LINES * self.typical_line_cases[sku])

    def season_factor(self, category: str, d: date) -> float:
        f = 1.0
        if category == config.SUMMER_PEAK_CATEGORY and d.month in config.SUMMER_PEAK_MONTHS:
            f *= config.SUMMER_PEAK_FACTOR
        if d.month == 8:
            f *= config.AUGUST_DEMAND_FACTOR
        return f

    def daily_demand(self, dc: str, sku: str, d: date) -> float:
        return self.base_demand[dc][sku] * self.season_factor(self.products[sku].category, d)

    def network_demand(self, sku: str, d: date) -> float:
        return sum(self.daily_demand(dc, sku, d) for dc in config.DC_IDS)

    def peak_build_up(self, sku: str, d: date) -> bool:
        """Before the summer peak, stock for that category is built up."""
        return self.products[sku].category == config.SUMMER_PEAK_CATEGORY and d.month in {5, 6}

    # -- production -------------------------------------------------------

    def produce(self, sku: str, produced_at: datetime, qty: int) -> str:
        p = self.products[sku]
        lot = f"L{produced_at:%y%j}-{sku}"
        best_before = produced_at.date() + timedelta(days=p.shelf_life_days)
        self.rows["production_batches"].append(
            (lot, sku, config.PLANT_ID, ts(produced_at), qty, best_before.isoformat())
        )
        self.stock.add(config.PLANT_ID, sku, lot, qty)
        if (
            sku == self.bad_batch_sku
            and self.bad_batch_lot is None
            and config.BAD_BATCH_PRODUCED <= produced_at.date() < config.BAD_BATCH_PRODUCED + timedelta(days=7)
        ):
            self.bad_batch_lot = lot
        return lot

    def opening_stock(self) -> None:
        """Stock on hand on 1 Jan 2025, made by the plant in December 2024."""
        start = config.START_DATE
        for p in self.md.products:
            produced = at(date(2024, 12, self.rng.randint(2, 20)), self.rng.randint(6, 14))
            network = self.network_demand(p.sku, start)
            total = round(network * config.PLANT_TARGET_DAYS_OF_COVER) + sum(
                round(self.dc_target(dc, p.sku, start)) for dc in config.DC_IDS
            )
            self.produce(p.sku, produced, max(total, p.cases_per_layer))
            for dc in config.DC_IDS:
                qty = round(self.dc_target(dc, p.sku, start))
                taken = self.stock.take(config.PLANT_ID, p.sku, qty)
                for lot_taken, n in taken:
                    self.stock.add(dc, p.sku, lot_taken, n)

    def run_production(self, d: date) -> None:
        if d.weekday() not in config.PLANT_PRODUCTION_WEEKDAYS:
            return
        for p in self.md.products:
            if self.production_weekday[p.sku] != d.weekday():
                continue
            look_ahead = d + timedelta(days=10)
            target = self.network_demand(p.sku, look_ahead) * config.PLANT_TARGET_DAYS_OF_COVER
            if self.peak_build_up(p.sku, d):
                target *= 1.6
            qty = round(target - self.stock.total(config.PLANT_ID, p.sku))
            if qty >= p.cases_per_layer:
                qty = math.ceil(qty / p.cases_per_layer) * p.cases_per_layer
                self.produce(p.sku, at(d, self.rng.randint(6, 14), self.rng.randint(0, 59)), qty)

    # -- pallets ----------------------------------------------------------

    def build_pallets(self, items: list[Item]) -> list[Pallet]:
        """Full pallets of one product first, then the leftovers on mixed pallets."""
        pallets: list[Pallet] = []
        leftovers: list[Item] = []
        for it in items:
            p = self.products[it.sku]
            full, rest = divmod(it.qty, p.cases_per_pallet)
            for _ in range(full):
                pallets.append(self._make_pallet([Item(**{**it.__dict__, "qty": p.cases_per_pallet})]))
            if rest:
                leftovers.append(Item(**{**it.__dict__, "qty": rest}))

        current: list[Item] = []
        fill = 0.0
        for it in leftovers:
            p = self.products[it.sku]
            share = it.qty / p.cases_per_pallet
            if current and (fill + share > 1.0 or self._height(current + [it]) > config.MAX_PALLET_BUILD_HEIGHT_CM):
                pallets.append(self._make_pallet(current))
                current, fill = [], 0.0
            current.append(it)
            fill += share
        if current:
            pallets.append(self._make_pallet(current))
        return pallets

    def _height(self, items: list[Item]) -> int:
        layers_cm = 0
        for it in items:
            p = self.products[it.sku]
            layers_cm += math.ceil(it.qty / p.cases_per_layer) * p.case_height_cm
        return PALLET_BASE_CM + layers_cm

    def _make_pallet(self, items: list[Item]) -> Pallet:
        weight = PALLET_TARE_KG + sum(it.qty * self.products[it.sku].case_weight_kg for it in items)
        return Pallet(self.new_sscc(), items, self._height(items), round(weight, 1))

    def write_pallet(self, pallet: Pallet, shipment_id: str, site: str, built_at: datetime) -> None:
        kinds = {it.sku for it in pallet.items}
        self.rows["pallets"].append(
            (
                pallet.sscc,
                shipment_id,
                site,
                ts(built_at),
                "single-SKU" if len(kinds) == 1 else "mixed",
                pallet.height_cm,
                pallet.weight_kg,
            )
        )
        for it in pallet.items:
            p = self.products[it.sku]
            best_before = self.best_before(it.lot, p)
            self.rows["pallet_contents"].append(
                (
                    pallet.sscc,
                    it.sku,
                    it.order_line_id,
                    it.transfer_line_id,
                    it.return_line_id,
                    it.qty,
                    it.lot,
                    best_before,
                )
            )

    def best_before(self, lot: str, p: Product) -> str:
        produced = datetime.strptime(lot[1:6], "%y%j").date()
        return (produced + timedelta(days=p.shelf_life_days)).isoformat()

    def pallet_event(self, sscc: str, event_type: str, when: datetime, location: str, scanned_by: str) -> None:
        if happened(when):
            self.rows["pallet_events"].append((self.new_id("PE", 8), sscc, event_type, ts(when), location, scanned_by))

    def shipment_event(
        self, shipment_id: str, event_type: str, when: datetime, location: str, reason: str | None = None
    ) -> None:
        if happened(when):
            self.rows["shipment_events"].append(
                (self.new_id("SE", 7), shipment_id, event_type, ts(when), reason, location)
            )

    def scanner(self, site: str) -> str:
        return f"{site}-RF{self.rng.randint(1, 12):02d}"

    # -- docks ------------------------------------------------------------

    def door(self, site: str) -> str:
        return f"D{self.rng.randint(1, self.sites[site].dock_doors):02d}"

    def outbound_dock(self, site: str, carrier_id: str, slot: datetime, shipment_id: str, n_pallets: int) -> datetime:
        """Book a loading slot, play out the truck's arrival and loading. Returns the departure time."""
        carrier = self.carriers[carrier_id]
        no_show = carrier.no_show_prob * (config.AUGUST_NO_SHOW_FACTOR if slot.month == 8 else 1.0)
        if self.rng.random() < no_show:
            self._appointment(
                site, "outbound", shipment_id, carrier_id, slot, None, None, None, "no-show", "truck late"
            )
            slot = slot + timedelta(hours=self.rng.choice([2, 3]))

        arrival = slot + timedelta(minutes=self.rng.randint(-25, 15))
        reason = None
        if self.rng.random() < 0.08:
            arrival += timedelta(minutes=self.rng.randint(30, 120))
            reason = "truck late"
        wait = self.rng.randint(5, 20)
        if site == config.SLOW_DOCK_DC and arrival.hour in config.SLOW_DOCK_HOURS:
            wait += self.rng.randint(*config.SLOW_DOCK_EXTRA_WAIT_MIN)
            reason = reason or "dock busy"
        if self.rng.random() < 0.03:
            wait += self.rng.randint(30, 90)
            reason = "goods not ready"
        dock_start = arrival + timedelta(minutes=wait)
        dock_end = dock_start + timedelta(minutes=min(20 + 2 * n_pallets, 90))
        self._appointment(
            site, "outbound", shipment_id, carrier_id, slot, arrival, dock_start, dock_end, "completed", reason
        )
        return dock_end + timedelta(minutes=10)

    def inbound_dock(
        self, site: str, carrier_id: str, planned: datetime, arrival: datetime, shipment_id: str, n_pallets: int
    ) -> datetime:
        """Unloading at a site. Returns the time the goods are received."""
        slot = planned.replace(minute=0, second=0)
        reason = "truck late" if arrival > slot + timedelta(hours=1) else None
        wait = self.rng.randint(5, 25)
        if site == config.SLOW_DOCK_DC and arrival.hour in config.SLOW_DOCK_HOURS:
            wait += self.rng.randint(*config.SLOW_DOCK_EXTRA_WAIT_MIN)
            reason = reason or "dock busy"
        dock_start = arrival + timedelta(minutes=wait)
        dock_end = dock_start + timedelta(minutes=min(20 + 2 * n_pallets, 90))
        status = "completed" if happened(dock_end) else "booked"
        self._appointment(site, "inbound", shipment_id, carrier_id, slot, arrival, dock_start, dock_end, status, reason)
        return dock_end

    def _appointment(
        self, site, direction, shipment_id, carrier_id, slot, checked_in, dock_start, dock_end, status, reason
    ) -> None:
        if not happened(checked_in):
            checked_in = dock_start = dock_end = None
            if status == "completed":
                status = "booked"
                reason = None
        elif not happened(dock_end):
            dock_end = None
            status = "in progress"
        self.rows["dock_appointments"].append(
            (
                self.new_id("DA", 7),
                site,
                self.door(site),
                direction,
                shipment_id,
                carrier_id,
                self.truck_plate(),
                ts(slot),
                ts(slot + timedelta(hours=1)),
                ts(checked_in),
                ts(dock_start),
                ts(dock_end),
                status,
                reason,
            )
        )

    # -- daily steps ------------------------------------------------------

    def put_away_arrivals(self, d: date) -> None:
        for site, sku, lot, qty in self.arrivals.pop(d, []):
            self.stock.add(site, sku, lot, qty)

    def place_orders(self, d: date) -> None:
        weekday_factor = config.WEEKDAY_FACTOR[d.weekday()]
        if weekday_factor == 0:
            return
        month_factor = config.AUGUST_DEMAND_FACTOR if d.month == 8 else 1.0
        weights = [p.popularity * self.season_factor(p.category, d) for p in self.md.products]
        ship_day = next_operating_day(d)
        for c in self.md.customers:
            if self.rng.random() >= config.DAILY_ORDER_PROB * c.order_frequency * weekday_factor * month_factor:
                continue
            n_lines = self.rng.randint(config.ORDER_LINES_MIN, config.ORDER_LINES_MAX)
            skus = list(dict.fromkeys(self.rng.choices(self.popularity_skus, weights, k=n_lines)))
            order_id = self.new_id("O", 6)
            lines = [OrderLine(self.new_id("OL", 7), sku, self.line_quantity(sku)) for sku in skus]
            order = Order(
                order_id=order_id,
                customer=c,
                order_date=at(d, self.rng.randint(8, 16), self.rng.randint(0, 59)),
                ship_day=ship_day,
                requested_delivery=next_operating_day(ship_day),
                cutoff=at(ship_day, 20),
                lines=lines,
            )
            self.orders.append(order)
            self.pending_orders.append(order)

    def ship_orders(self, d: date) -> None:
        due = [o for o in self.pending_orders if o.ship_day == d]
        self.pending_orders = [o for o in self.pending_orders if o.ship_day != d]
        for order in due:
            dc = order.customer.home_dc
            items: list[Item] = []
            for line in order.lines:
                taken = self.stock.take(dc, line.sku, line.qty_ordered)
                line.qty_shipped = sum(n for _, n in taken)
                line.lots = taken
                items += [Item(line.sku, lot, n, order_line_id=line.order_line_id) for lot, n in taken]
            if not items:
                order.status = "cancelled"
                continue
            order.status = "picked"
            self.dispatch_order(order, items)

    def dispatch_order(self, order: Order, items: list[Item]) -> None:
        c = order.customer
        dc = c.home_dc
        pallets = self.build_pallets(items)
        weight = sum(p.weight_kg for p in pallets)
        if len(pallets) == 1 and weight < 400:
            carrier_id = self.rng.choice(self.parcel_carriers)
        else:
            mode = "LTL" if len(pallets) <= 8 else "FTL"
            pool = self.ltl_carriers if mode == "LTL" else self.ftl_carriers
            carrier_id = (
                self.preferred_carrier[(c.customer_id, mode)] if self.rng.random() < 0.8 else self.rng.choice(pool)
            )
        carrier = self.carriers[carrier_id]

        shipment_id = self.new_id("S", 6)
        slot = at(order.ship_day, self.rng.randint(15, 19))
        planned_ship = slot + timedelta(hours=1)
        win_start = datetime.combine(order.requested_delivery, time.fromisoformat(c.delivery_window_start))
        win_end = datetime.combine(order.requested_delivery, time.fromisoformat(c.delivery_window_end))
        planned_delivery = win_start + timedelta(minutes=self.rng.randint(0, 90))

        for pallet in pallets:
            built = slot - timedelta(minutes=self.rng.randint(60, 240))
            self.write_pallet(pallet, shipment_id, dc, built)
            self.pallet_event(pallet.sscc, "built", built, f"{dc} packing", self.scanner(dc))
            self.pallet_event(
                pallet.sscc,
                "staged",
                slot - timedelta(minutes=self.rng.randint(10, 50)),
                f"{dc} staging lane",
                self.scanner(dc),
            )

        depart = self.outbound_dock(dc, carrier_id, slot, shipment_id, len(pallets))
        for pallet in pallets:
            self.pallet_event(
                pallet.sscc,
                "loaded",
                depart - timedelta(minutes=self.rng.randint(12, 40)),
                f"{dc} dock",
                self.scanner(dc),
            )
            self.pallet_event(pallet.sscc, "departed", depart, dc, self.scanner(dc))
        self.shipment_event(shipment_id, "departed", depart, self.sites[dc].city)

        # transit and delivery: the truck aims to arrive so unloading ends at the planned time
        unload = timedelta(minutes=self.rng.randint(15, 45))
        arrival = max(planned_delivery - unload, depart + timedelta(hours=self.customer_transit_h[c.customer_id]))
        delay_reason = None
        if arrival + unload > planned_delivery + timedelta(minutes=30):
            delay_reason = "ferry delay" if c.region in ISLAND_REGIONS else "late loading"
        late_prob = carrier.late_prob
        if carrier_id == config.MONDAY_LATE_CARRIER and arrival.weekday() == 0:
            late_prob = config.MONDAY_LATE_PROB
        if self.rng.random() < late_prob:
            arrival += timedelta(minutes=self.rng.randint(45, 300))
            delay_reason = self.rng.choice(["traffic", "truck breakdown", "driver hours"])
        if delay_reason:
            self.shipment_event(shipment_id, "delayed", depart + (arrival - depart) / 2, "en route", delay_reason)
        delivered = arrival + unload
        self.shipment_event(shipment_id, "arrived", arrival, c.region)
        self.shipment_event(shipment_id, "delivered", delivered, c.region)

        status = "delivered" if happened(delivered) else ("in transit" if happened(depart) else "planned")
        self.rows["shipments"].append(
            (
                shipment_id,
                "customer delivery",
                order.order_id,
                None,
                None,
                carrier_id,
                dc,
                None,
                None,
                ts(planned_ship),
                ts(depart) if happened(depart) else None,
                ts(planned_delivery),
                ts(delivered) if happened(delivered) else None,
                status,
            )
        )
        order.status = {"delivered": "delivered", "in transit": "shipped", "planned": "picked"}[status]

        # refusals at delivery
        very_late = delivered > win_end + timedelta(hours=2)
        refused: list[tuple[Pallet, str]] = []
        for pallet in pallets:
            reason = None
            if self.rng.random() < carrier.damage_prob:
                self.pallet_event(
                    pallet.sscc, "damaged", delivered - timedelta(minutes=5), c.region, f"DRV-{carrier_id}"
                )
                if self.rng.random() < config.REFUSE_IF_DAMAGED_PROB:
                    reason = "damaged"
            if reason is None and pallet.height_cm > c.max_pallet_height_cm:
                strict = c.max_pallet_height_cm < config.MAX_PALLET_HEIGHT_CM_DEFAULT
                prob = config.REFUSE_IF_TOO_TALL_STRICT_PROB if strict else config.REFUSE_IF_TOO_TALL_PROB
                if self.rng.random() < prob:
                    reason = "pallet too tall"
            if reason is None and very_late and self.rng.random() < config.REFUSE_IF_VERY_LATE_PROB:
                reason = "delivered late"
            if reason:
                refused.append((pallet, reason))
                self.pallet_event(pallet.sscc, "rejected", delivered, c.region, f"DRV-{carrier_id}")
            else:
                self.pallet_event(pallet.sscc, "received", delivered, c.region, f"DRV-{carrier_id}")

        if refused and happened(delivered):
            self.refused_return(order, refused, delivered)

        # later returns
        if happened(delivered):
            bad = [ln for ln in order.lines if any(lot == self.bad_batch_lot for lot, _ in ln.lots)]
            if bad and self.rng.random() < config.BAD_BATCH_RETURN_PROB:
                when = delivered.date() + timedelta(days=self.rng.randint(7, 30))
                self.scheduled_returns[when].append((order, bad, "quality complaint"))
            elif self.rng.random() < config.LATER_RETURN_PROB_PER_ORDER:
                when = delivered.date() + timedelta(days=self.rng.randint(*config.LATER_RETURN_DELAY_DAYS))
                lines = [ln for ln in order.lines if ln.qty_shipped > 0]
                picked = self.rng.sample(lines, k=min(len(lines), self.rng.randint(1, 2)))
                reason = self.rng.choice([r for r in config.RETURN_REASONS if r != "quality complaint"])
                self.scheduled_returns[when].append((order, picked, reason))

    # -- returns ----------------------------------------------------------

    def inspect(self, reason: str) -> tuple[str, str]:
        r = self.rng.random()
        if reason == "damaged":
            return ("damaged product", "destroy") if r < 0.6 else ("damaged packaging", "rework")
        if reason == "quality complaint":
            return "damaged product", "destroy"
        if reason == "short dated":
            return ("expired", "destroy") if r < 0.5 else ("sellable", "restock")
        return "sellable", "restock"

    def refused_return(self, order: Order, refused: list[tuple[Pallet, str]], delivered: datetime) -> None:
        """Refused pallets go back on the same truck to the DC they came from."""
        c = order.customer
        dc = c.home_dc
        return_id = self.new_id("R", 5)
        reasons = [r for _, r in refused]
        main_reason = max(sorted(set(reasons)), key=reasons.count)
        back_at_dc = delivered + timedelta(hours=self.customer_transit_h[c.customer_id] + self.rng.uniform(0.5, 3))
        inspected = back_at_dc + timedelta(hours=self.rng.uniform(2, 20))
        status = "closed" if happened(inspected) else ("received" if happened(back_at_dc) else "requested")
        self.rows["customer_returns"].append(
            (return_id, c.customer_id, dc, order.order_id, "refused at delivery", ts(delivered), main_reason, status)
        )
        for pallet, reason in refused:
            self.pallet_event(pallet.sscc, "received", back_at_dc, f"{dc} returns dock", self.scanner(dc))
            for it in pallet.items:
                condition, disposition = self.inspect(reason)
                done = happened(inspected)
                self.rows["customer_return_lines"].append(
                    (
                        self.new_id("RL", 6),
                        return_id,
                        it.order_line_id,
                        it.sku,
                        it.lot,
                        it.qty,
                        it.qty if happened(back_at_dc) else 0,
                        condition if done else None,
                        disposition if done else None,
                        ts(inspected) if done else None,
                    )
                )
                if done and disposition == "restock":
                    self.arrivals[inspected.date() + timedelta(days=1)].append((dc, it.sku, it.lot, it.qty))

    def process_later_returns(self, d: date) -> None:
        for order, lines, reason in self.scheduled_returns.pop(d, []):
            if not is_operating_day(d):
                self.scheduled_returns[next_operating_day(d)].append((order, lines, reason))
                continue
            self.later_return(order, lines, reason, d)

    def later_return(self, order: Order, lines: list[OrderLine], reason: str, d: date) -> None:
        c = order.customer
        dc = c.home_dc
        return_id = self.new_id("R", 5)
        requested = at(d, self.rng.randint(8, 15), self.rng.randint(0, 59))

        pickup_day = d
        for _ in range(self.rng.randint(2, 4)):
            pickup_day = next_operating_day(pickup_day)
        carrier_id = self.preferred_carrier[(c.customer_id, "LTL")]
        planned_ship = at(pickup_day, 10)
        actual_ship = planned_ship + timedelta(minutes=self.rng.randint(-20, 90))
        planned_arrival = planned_ship + timedelta(hours=self.customer_transit_h[c.customer_id])
        arrival = actual_ship + timedelta(hours=self.customer_transit_h[c.customer_id] + self.rng.uniform(0, 1.5))
        shipment_id = self.new_id("S", 6)

        items: list[Item] = []
        line_rows = []
        for ln in lines:
            if not ln.lots:
                continue
            lot, shipped_from_lot = ln.lots[0]
            if reason == "quality complaint":
                lot, shipped_from_lot = next((x for x in ln.lots if x[0] == self.bad_batch_lot), ln.lots[0])
                qty = max(1, round(shipped_from_lot * self.rng.uniform(0.3, 1.0)))
            else:
                qty = self.rng.randint(1, min(shipped_from_lot, 20))
            rl_id = self.new_id("RL", 6)
            items.append(Item(ln.sku, lot, qty, return_line_id=rl_id))
            line_rows.append((rl_id, ln, lot, qty))
        if not items:
            return

        pallets = self.build_pallets(items)
        received = inspected = None
        if happened(actual_ship):
            received = self.inbound_dock(dc, carrier_id, planned_arrival, arrival, shipment_id, len(pallets))
            inspected = received + timedelta(hours=self.rng.uniform(4, 48))

        if happened(inspected):
            status = "closed"
        elif happened(received):
            status = "received"
        elif happened(actual_ship):
            status = "collected"
        elif happened(requested + timedelta(days=1)):
            status = "approved"
        else:
            status = "requested"
        self.rows["customer_returns"].append(
            (return_id, c.customer_id, dc, order.order_id, "later return", ts(requested), reason, status)
        )
        for rl_id, ln, lot, qty in line_rows:
            condition, disposition = self.inspect(reason)
            done = happened(inspected)
            self.rows["customer_return_lines"].append(
                (
                    rl_id,
                    return_id,
                    ln.order_line_id,
                    ln.sku,
                    lot,
                    qty,
                    qty if happened(received) else 0,
                    condition if done else None,
                    disposition if done else None,
                    ts(inspected) if done else None,
                )
            )
            if done and disposition == "restock":
                self.arrivals[inspected.date() + timedelta(days=1)].append((dc, ln.sku, lot, qty))

        if not happened(actual_ship):
            return  # not collected before the end of the year: no shipment yet
        shipment_status = "delivered" if happened(arrival) else "in transit"
        self.rows["shipments"].append(
            (
                shipment_id,
                "customer return",
                None,
                None,
                return_id,
                carrier_id,
                None,
                c.customer_id,
                dc,
                ts(planned_ship),
                ts(actual_ship),
                ts(planned_arrival),
                ts(arrival) if happened(arrival) else None,
                shipment_status,
            )
        )
        self.shipment_event(shipment_id, "departed", actual_ship, c.region)
        self.shipment_event(shipment_id, "delivered", arrival, self.sites[dc].city)
        if happened(received):
            for pallet in pallets:
                built = received + timedelta(minutes=self.rng.randint(10, 60))
                self.write_pallet(pallet, shipment_id, dc, built)
                self.pallet_event(pallet.sscc, "built", built, f"{dc} returns area", self.scanner(dc))

    # -- transfers --------------------------------------------------------

    def plan_plant_transfers(self, d: date) -> None:
        """On plant shipment days, top up every DC to its target cover."""
        if d.weekday() not in config.PLANT_SHIPMENT_WEEKDAYS:
            return
        for dc in config.DC_IDS:
            groups: dict[str, list[TransferLine]] = defaultdict(list)
            for p in self.md.products:
                target = self.dc_target(dc, p.sku, d + timedelta(days=5))
                build_up = self.peak_build_up(p.sku, d)
                if build_up:
                    target *= 1.6
                position = self.stock.total(dc, p.sku) + self.in_transit[(dc, p.sku)]
                need = target - position
                if need < p.cases_per_layer:
                    continue
                qty = min(
                    math.ceil(need / p.cases_per_layer) * p.cases_per_layer, self.stock.total(config.PLANT_ID, p.sku)
                )
                if qty <= 0:
                    continue
                reason = "peak build-up" if build_up else "replenishment"
                groups[reason].append(TransferLine(self.new_id("TL", 6), p.sku, qty))
            for reason, lines in groups.items():
                self.create_transfer(config.PLANT_ID, dc, at(d, 6, self.rng.randint(0, 30)), d, reason, lines)

    def plan_rebalancing(self, d: date) -> None:
        """A DC that is running low asks another DC with plenty of stock."""
        groups: dict[tuple[str, str], list[TransferLine]] = defaultdict(list)
        for dc in config.DC_IDS:
            for p in self.md.products:
                daily = self.daily_demand(dc, p.sku, d)
                if daily <= 0:
                    continue
                position = self.stock.total(dc, p.sku) + self.in_transit[(dc, p.sku)]
                if position >= max(daily * config.REBALANCE_TRIGGER_DAYS, self.typical_line_cases[p.sku]):
                    continue
                best, best_cover = None, config.REBALANCE_SOURCE_MIN_DAYS
                for src in config.DC_IDS:
                    src_daily = self.daily_demand(src, p.sku, d)
                    if src == dc or src_daily <= 0:
                        continue
                    cover = self.stock.total(src, p.sku) / src_daily
                    if cover > best_cover:
                        best, best_cover = src, cover
                if best is None:
                    continue
                spare = self.stock.total(best, p.sku) - self.daily_demand(best, p.sku, d) * 8
                qty = min(round(daily * config.REBALANCE_QTY_DAYS), round(spare))
                if qty >= p.cases_per_layer:
                    groups[(best, dc)].append(TransferLine(self.new_id("TL", 6), p.sku, qty))
                    self.in_transit[(dc, p.sku)] += qty  # reserve so we do not ask twice
        ship_day = next_operating_day(d)
        for (src, dst), lines in groups.items():
            self.create_transfer(
                src, dst, at(d, self.rng.randint(9, 17)), ship_day, "rebalancing", lines, reserved=True
            )

    def create_transfer(
        self,
        src: str,
        dst: str,
        created: datetime,
        ship_day: date,
        reason: str,
        lines: list[TransferLine],
        reserved: bool = False,
    ) -> None:
        transit = config.TRANSIT_HOURS[frozenset({src, dst})]
        t = Transfer(
            transfer_id=self.new_id("T", 5),
            source=src,
            dest=dst,
            created_at=created,
            ship_day=ship_day,
            requested_arrival=(at(ship_day, 14) + timedelta(hours=transit)).date(),
            reason=reason,
            lines=lines,
        )
        if not reserved:
            for ln in lines:
                self.in_transit[(dst, ln.sku)] += ln.qty_requested
        self.transfers.append(t)
        self.pending_transfers.append(t)

    def ship_transfers(self, d: date) -> None:
        due = [t for t in self.pending_transfers if t.ship_day == d]
        self.pending_transfers = [t for t in self.pending_transfers if t.ship_day != d]
        for t in due:
            items: list[Item] = []
            for ln in t.lines:
                taken = self.stock.take(t.source, ln.sku, ln.qty_requested)
                ln.qty_shipped = sum(n for _, n in taken)
                self.in_transit[(t.dest, ln.sku)] -= ln.qty_requested - ln.qty_shipped  # shortfall never comes
                items += [Item(ln.sku, lot, n, transfer_line_id=ln.transfer_line_id) for lot, n in taken]
            if not items:
                t.status = "cancelled"
                continue
            t.status = "shipped"
            pallets = self.build_pallets(items)
            for i in range(0, len(pallets), config.TRUCK_CAPACITY_PALLETS):
                self.dispatch_transfer_truck(t, pallets[i : i + config.TRUCK_CAPACITY_PALLETS], d)

    def dispatch_transfer_truck(self, t: Transfer, pallets: list[Pallet], d: date) -> None:
        carrier_id = self.rng.choice(self.ftl_carriers)
        carrier = self.carriers[carrier_id]
        shipment_id = self.new_id("S", 6)
        slot = at(d, self.rng.randint(7, 13))
        transit = timedelta(hours=config.TRANSIT_HOURS[frozenset({t.source, t.dest})])

        for pallet in pallets:
            built = slot - timedelta(minutes=self.rng.randint(60, 300))
            self.write_pallet(pallet, shipment_id, t.source, built)
            self.pallet_event(pallet.sscc, "built", built, f"{t.source} packing", self.scanner(t.source))
            self.pallet_event(
                pallet.sscc,
                "staged",
                slot - timedelta(minutes=self.rng.randint(10, 50)),
                f"{t.source} staging lane",
                self.scanner(t.source),
            )

        planned_ship = slot + timedelta(hours=1)
        planned_arrival = planned_ship + transit
        depart = self.outbound_dock(t.source, carrier_id, slot, shipment_id, len(pallets))
        for pallet in pallets:
            self.pallet_event(
                pallet.sscc,
                "loaded",
                depart - timedelta(minutes=self.rng.randint(12, 40)),
                f"{t.source} dock",
                self.scanner(t.source),
            )
            self.pallet_event(pallet.sscc, "departed", depart, t.source, self.scanner(t.source))
        self.shipment_event(shipment_id, "departed", depart, self.sites[t.source].city)

        arrival = depart + transit
        if self.rng.random() < carrier.late_prob:
            arrival += timedelta(minutes=self.rng.randint(30, 240))
            self.shipment_event(
                shipment_id,
                "delayed",
                depart + transit / 2,
                "en route",
                self.rng.choice(["traffic", "truck breakdown", "driver hours"]),
            )

        # Pattern 4: inbound to the stockout DC with that category is held, so the DC runs out
        holds_category = any(
            self.products[it.sku].category == config.STOCKOUT_CATEGORY for p in pallets for it in p.items
        )
        if (
            t.dest == config.STOCKOUT_DC
            and holds_category
            and config.STOCKOUT_START - timedelta(days=config.STOCKOUT_HOLD_DAYS_BEFORE) <= d <= config.STOCKOUT_END
        ):
            released = at(config.STOCKOUT_END + timedelta(days=1), 6, self.rng.randint(0, 59))
            self.shipment_event(
                shipment_id, "delayed", depart + timedelta(minutes=30), self.sites[t.source].city, "held at origin"
            )
            arrival = released + transit

        received = self.inbound_dock(t.dest, carrier_id, planned_arrival, arrival, shipment_id, len(pallets))
        self.shipment_event(shipment_id, "arrived", arrival, self.sites[t.dest].city)
        self.shipment_event(shipment_id, "delivered", received, self.sites[t.dest].city)

        lossy = (t.source, t.dest) == config.LOSSY_LANE
        lines = {ln.transfer_line_id: ln for ln in t.lines}
        for pallet in pallets:
            self.pallet_event(pallet.sscc, "received", received, f"{t.dest} inbound dock", self.scanner(t.dest))
            for it in pallet.items:
                qty = it.qty
                if lossy and self.rng.random() < config.LOSSY_LANE_LOSS_PROB:
                    qty -= self.rng.randint(1, min(5, qty))
                ln = lines[it.transfer_line_id]
                self.in_transit[(t.dest, it.sku)] -= it.qty
                if happened(received):
                    ln.qty_received += qty
                    self.arrivals[received.date() + timedelta(days=1)].append((t.dest, it.sku, it.lot, qty))

        status = "delivered" if happened(received) else ("in transit" if happened(depart) else "planned")
        self.rows["shipments"].append(
            (
                shipment_id,
                "stock transfer",
                None,
                t.transfer_id,
                None,
                carrier_id,
                t.source,
                None,
                t.dest,
                ts(planned_ship),
                ts(depart) if happened(depart) else None,
                ts(planned_arrival),
                ts(received) if happened(received) else None,
                status,
            )
        )
        if happened(received) and all(ln.qty_received > 0 or ln.qty_shipped == 0 for ln in t.lines):
            t.status = "received"

    # -- inventory --------------------------------------------------------

    def snapshot(self, d: date) -> None:
        allocated: dict[tuple[str, str], int] = defaultdict(int)
        for o in self.pending_orders:
            for ln in o.lines:
                allocated[(o.customer.home_dc, ln.sku)] += ln.qty_ordered
        for t in self.pending_transfers:
            for ln in t.lines:
                allocated[(t.source, ln.sku)] += ln.qty_requested
        day = d.isoformat()
        for site in self.sites:
            for p in self.md.products:
                on_hand = self.stock.total(site, p.sku)
                self.rows["inventory"].append((day, site, p.sku, on_hand, min(on_hand, allocated[(site, p.sku)])))

    # -- main loop --------------------------------------------------------

    def run(self) -> None:
        self.opening_stock()
        d = config.START_DATE
        while d <= config.END_DATE:
            self.put_away_arrivals(d)
            if is_operating_day(d):
                self.plan_plant_transfers(d)
                self.ship_transfers(d)
                self.run_production(d)  # after shipping: today's production ships from the next shipment day
                self.ship_orders(d)
                self.process_later_returns(d)
                self.place_orders(d)
                self.plan_rebalancing(d)
            else:
                # returns due on a closed day move to the next operating day
                for item in self.scheduled_returns.pop(d, []):
                    self.scheduled_returns[next_operating_day(d)].append(item)
            self.snapshot(d)
            d += timedelta(days=1)
        self._write_orders_and_transfers()

    def _write_orders_and_transfers(self) -> None:
        for o in self.orders:
            self.rows["orders"].append(
                (
                    o.order_id,
                    o.customer.customer_id,
                    o.customer.home_dc,
                    ts(o.order_date),
                    o.requested_delivery.isoformat(),
                    ts(o.cutoff),
                    o.status,
                )
            )
            for ln in o.lines:
                self.rows["order_lines"].append((ln.order_line_id, o.order_id, ln.sku, ln.qty_ordered, ln.qty_shipped))
        for t in self.transfers:
            self.rows["stock_transfers"].append(
                (t.transfer_id, t.source, t.dest, ts(t.created_at), t.requested_arrival.isoformat(), t.reason, t.status)
            )
            for ln in t.lines:
                self.rows["stock_transfer_lines"].append(
                    (ln.transfer_line_id, t.transfer_id, ln.sku, ln.qty_requested, ln.qty_shipped, ln.qty_received)
                )


def simulate(md: MasterData, rng: random.Random) -> dict[str, list[tuple]]:
    sim = Simulation(md, rng)
    sim.run()
    return sim.rows
