"""Master data: sites (plant and DCs), products, customers and carriers.

Each record has the columns that go into the database, plus two hidden
behaviour fields the simulation uses to create demand (a product's
popularity and a customer's order frequency). Those two are never written
to the database: the data assistant has to discover them from the orders.
"""

import random
import sqlite3
from dataclasses import dataclass

from . import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gs1_check_digit(digits: str) -> str:
    """GS1 mod-10 check digit for GTINs and SSCCs.

    Weights 3 and 1 alternate starting from the rightmost digit.
    """
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return str((10 - total % 10) % 10)


def make_gtin13(item_number: int) -> str:
    body = f"{config.GS1_COMPANY_PREFIX}{item_number:05d}"
    return body + gs1_check_digit(body)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Product:
    sku: str
    product_name: str
    brand: str
    category: str
    gtin: str
    units_per_case: int
    cases_per_layer: int
    cases_per_pallet: int
    case_weight_kg: float
    case_height_cm: int
    shelf_life_days: int
    # hidden behaviour, not stored
    popularity: float  # relative demand weight within the catalogue


@dataclass
class Customer:
    customer_id: str
    customer_name: str
    region: str
    country: str
    home_dc: str
    delivery_window_start: str
    delivery_window_end: str
    max_pallet_height_cm: int
    # hidden behaviour, not stored
    order_frequency: float  # multiplier on the daily order probability


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

# category -> (brands, {product type: sizes}, units per case range, case kg range, shelf life days)
CATALOGUE = {
    "Laundry": (
        ["Aqualis", "Brightex", "Linea"],
        {
            "Liquid detergent": ["1L", "2L", "3L"],
            "Powder detergent": ["1.5kg", "3kg", "5kg"],
            "Laundry pods": ["20ct", "40ct", "60ct"],
            "Fabric softener": ["1L", "2L"],
            "Stain remover": ["500ml", "1L"],
        },
        (4, 12),
        (6.0, 14.0),
        900,
    ),
    "Baby care": (
        ["Nidi", "Softly"],
        {
            "Diapers size 3": ["single pack", "jumbo pack", "monthly box"],
            "Diapers size 4": ["single pack", "jumbo pack", "monthly box"],
            "Diapers size 5": ["single pack", "jumbo pack", "monthly box"],
            "Baby wipes": ["64ct", "4-pack", "12-pack"],
            "Baby pants": ["single pack", "jumbo pack"],
        },
        (2, 8),
        (4.0, 9.0),
        1095,
    ),
    "Hair care": (
        ["Kalli", "Mirto", "Velora"],
        {
            "Shampoo": ["250ml", "400ml", "750ml"],
            "Conditioner": ["200ml", "360ml"],
            "Hair mask": ["300ml"],
            "Anti-dandruff shampoo": ["250ml", "400ml"],
            "Hair spray": ["250ml", "400ml"],
        },
        (6, 12),
        (3.0, 8.0),
        900,
    ),
    "Oral care": (
        ["Dentiva", "Smilo"],
        {
            "Toothpaste": ["75ml", "100ml", "2-pack"],
            "Electric toothbrush heads": ["2-pack", "4-pack", "8-pack"],
            "Mouthwash": ["250ml", "500ml", "1L"],
            "Manual toothbrush": ["single", "2-pack", "4-pack"],
        },
        (12, 48),
        (2.0, 6.0),
        1095,
    ),
    "Personal care": (
        ["Thalassa", "Aura", "Zefyros"],
        {
            "Deodorant spray": ["150ml", "250ml"],
            "Shower gel": ["250ml", "500ml", "750ml"],
            "Razor blades": ["4-pack", "8-pack"],
            "Shaving foam": ["200ml", "300ml"],
            "Sunscreen": ["150ml", "200ml"],
        },
        (6, 24),
        (2.5, 7.0),
        900,
    ),
    "Home care": (
        ["Purita", "Casa Clean"],
        {
            "Dishwashing liquid": ["500ml", "1L", "1.5L"],
            "Dishwasher tablets": ["30ct", "60ct", "100ct"],
            "Surface cleaner": ["500ml", "1L"],
            "Kitchen towels": ["2-roll", "4-roll", "8-roll"],
            "Air freshener": ["300ml", "refill 2-pack"],
        },
        (6, 16),
        (4.0, 10.0),
        1095,
    ),
}

CATEGORY_CODES = {
    "Laundry": "LAU",
    "Baby care": "BAB",
    "Hair care": "HAI",
    "Oral care": "ORA",
    "Personal care": "PER",
    "Home care": "HOM",
}


def build_products(rng: random.Random) -> list[Product]:
    products: list[Product] = []
    item_number = 1
    for category, count in config.PRODUCTS_PER_CATEGORY.items():
        brands, types, units_range, kg_range, shelf_life = CATALOGUE[category]
        combos = [(b, t, s) for b in brands for t, sizes in types.items() for s in sizes]
        if len(combos) < count:
            raise ValueError(f"{category}: only {len(combos)} product combinations for {count} products")
        rng.shuffle(combos)
        for i, (brand, ptype, size) in enumerate(combos[:count], start=1):
            cases_per_layer = rng.choice([6, 8, 10, 12])
            layers = rng.choice([4, 5, 6, 7])
            products.append(
                Product(
                    sku=f"{CATEGORY_CODES[category]}-{i:03d}",
                    product_name=f"{brand} {ptype} {size}",
                    brand=brand,
                    category=category,
                    gtin=make_gtin13(item_number),
                    units_per_case=rng.randint(*units_range),
                    cases_per_layer=cases_per_layer,
                    cases_per_pallet=cases_per_layer * layers,
                    case_weight_kg=round(rng.uniform(*kg_range), 2),
                    case_height_cm=rng.choice([18, 20, 22, 25, 28]),
                    shelf_life_days=shelf_life,
                    # a few best sellers, a long tail of slower products
                    popularity=round(rng.paretovariate(1.5), 3),
                )
            )
            item_number += 1
    return products


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

REGIONS_BY_DC = {
    "ATH": ["Attica", "Central Greece", "Cyclades", "Dodecanese", "Crete", "North Aegean"],
    "THE": ["Central Macedonia", "Eastern Macedonia and Thrace", "Western Macedonia", "Thessaly"],
    "PAT": ["Western Greece", "Peloponnese", "Ionian Islands", "Epirus"],
}

NAME_STEMS = [
    "Agora",
    "Lefkos",
    "Galini",
    "Asteri",
    "Kyma",
    "Fotia",
    "Elia",
    "Pefko",
    "Delfini",
    "Ampeli",
    "Thymari",
    "Akti",
    "Rodi",
    "Kastro",
    "Limani",
    "Platanos",
    "Vrachos",
    "Anemos",
    "Ourania",
    "Kipos",
    "Selini",
    "Iris",
    "Nefeli",
    "Marina",
    "Orizon",
    "Zefyr",
    "Kerasi",
    "Melissa",
    "Dafni",
    "Lotos",
]
NAME_SUFFIXES = ["Markets", "Supermarkets", "Cash & Carry", "Wholesale", "Mini Markets", "Food Stores", "Distribution"]

DELIVERY_WINDOWS = [("06:00", "10:00"), ("07:00", "12:00"), ("08:00", "14:00"), ("10:00", "16:00"), ("06:00", "14:00")]


def build_customers(rng: random.Random) -> list[Customer]:
    names = [f"{stem} {suffix}" for stem in NAME_STEMS for suffix in NAME_SUFFIXES]
    rng.shuffle(names)

    dcs = list(config.CUSTOMER_SHARE_BY_DC)
    weights = list(config.CUSTOMER_SHARE_BY_DC.values())

    customers: list[Customer] = []
    for i in range(config.NUM_CUSTOMERS):
        home_dc = dcs[0] if i == config.STRICT_HEIGHT_CUSTOMER_INDEX else rng.choices(dcs, weights)[0]
        start, end = rng.choice(DELIVERY_WINDOWS)
        strict = i == config.STRICT_HEIGHT_CUSTOMER_INDEX
        customers.append(
            Customer(
                customer_id=f"CUS{i + 1:03d}",
                customer_name=names[i],
                region=rng.choice(REGIONS_BY_DC[home_dc]),
                country="Greece",
                home_dc=home_dc,
                delivery_window_start=start,
                delivery_window_end=end,
                max_pallet_height_cm=config.STRICT_HEIGHT_LIMIT_CM if strict else config.MAX_PALLET_HEIGHT_CM_DEFAULT,
                # strict customer is a big account, so the pattern is visible
                order_frequency=2.0 if strict else round(rng.uniform(0.4, 1.6), 2),
            )
        )
    return customers


# ---------------------------------------------------------------------------
# Write to the database
# ---------------------------------------------------------------------------


@dataclass
class MasterData:
    products: list[Product]
    customers: list[Customer]
    warehouses: list[config.Warehouse]
    carriers: list[config.Carrier]


def build_master_data(rng: random.Random) -> MasterData:
    return MasterData(
        products=build_products(rng),
        customers=build_customers(rng),
        warehouses=list(config.WAREHOUSES),
        carriers=list(config.CARRIERS),
    )


def insert_master_data(con: sqlite3.Connection, md: MasterData) -> None:
    con.executemany(
        "INSERT INTO warehouses VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (w.warehouse_id, w.name, w.city, "Greece", w.site_type, w.operator_type, w.operator_name)
            for w in md.warehouses
        ],
    )
    con.executemany(
        "INSERT INTO carriers VALUES (?, ?, ?, ?)",
        [(c.carrier_id, c.carrier_name, c.mode, c.cost_per_pallet) for c in md.carriers],
    )
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                p.sku,
                p.product_name,
                p.brand,
                p.category,
                p.gtin,
                p.units_per_case,
                p.cases_per_layer,
                p.cases_per_pallet,
                p.case_weight_kg,
                p.case_height_cm,
                p.shelf_life_days,
            )
            for p in md.products
        ],
    )
    con.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                c.customer_id,
                c.customer_name,
                c.region,
                c.country,
                c.home_dc,
                c.delivery_window_start,
                c.delivery_window_end,
                c.max_pallet_height_cm,
            )
            for c in md.customers
        ],
    )
