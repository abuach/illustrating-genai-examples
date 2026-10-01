"""Ask the database: the model writes SQL, a warehouse runs it.

The useful version of this is everywhere now: a business user types "Q3 revenue
by region?", a model turns it into SQL, the database runs the query, and a number
comes back. The dangerous failure isn't a crash. A crash is loud, and someone
fixes it. The dangerous failure is a query that *runs*, returns a plausible
number, and is silently wrong, because the model summed the catalog price instead
of the sold price, forgot that returned orders aren't revenue, or fanned a
per-order total out across line items. Nobody downstream can tell a wrong number
from a right one; they both look like a number.

This module builds a small but deliberately tricky retail warehouse in SQLite (no
files, no server), hand-writes a gold query for each business question, lets
gpt-oss:20b write its own SQL, and a deterministic checker sorts each model query
into one of three bins: correct, silent-wrong (ran but disagreed with gold), or
crash. The fix industry actually shipped is a governed semantic layer: instead of
free-form SQL, the model picks from a few *defined* metrics and dimensions, and a
compiler emits the one correct query. The model is gpt-oss:20b (EIS_MODEL); SQL
generation is nondeterministic, so capture SQL_AUDIT once and freeze the cells.
"""
import json
import random
import re
import sqlite3

import ollama

from genai.arch import EIS_MODEL

# ── The warehouse ─────────────────────────────────────────────────────────────
# A small retail schema with the traps real warehouses have. Two columns named
# `region` mean different things (where the customer is vs where the store is);
# `list_price` (catalog) is not `unit_price` (what it actually sold for); some
# orders are returned or cancelled and aren't revenue; `order_date` is not
# `ship_date`; and `orders.order_total` is a denormalized per-order figure that
# fans out if you sum it after joining to the line items.

REGIONS = ["West", "Central", "East"]
CATEGORIES = ["Hardware", "Software", "Service", "Accessory"]

SCHEMA_DDL = """\
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT,
    region      TEXT          -- the customer's billing region
);
CREATE TABLE stores (
    store_id INTEGER PRIMARY KEY,
    name     TEXT,
    region   TEXT             -- the store's physical region (NOT the customer's)
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT,
    category   TEXT,
    list_price REAL            -- catalog price; an item often sells for less
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER,
    store_id    INTEGER,
    order_date  TEXT,          -- when the order was placed (YYYY-MM-DD)
    ship_date   TEXT,          -- when it shipped; can fall in a later quarter
    status      TEXT,          -- 'completed', 'returned', or 'cancelled'
    order_total REAL           -- precomputed order total (a legacy column)
);
CREATE TABLE order_items (
    item_id    INTEGER PRIMARY KEY,
    order_id   INTEGER,
    product_id INTEGER,
    quantity   INTEGER,
    unit_price REAL            -- the price this line actually sold at
);
"""

# Plain-language data dictionary the chapter and the gold queries agree on:
# "revenue" means money actually collected = SUM(quantity * unit_price) over
# orders whose status is 'completed'; "region" means the customer's billing
# region; a quarter is sliced on order_date.


def build_warehouse() -> sqlite3.Connection:
    """Build and populate the in-memory warehouse deterministically (seed 7)."""
    rng = random.Random(7)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_DDL)

    customers = [(i, f"Cust{i:02d}", rng.choice(REGIONS)) for i in range(1, 13)]
    # Stores skew differently from customers so the two "region" groupings differ.
    stores = [(1, "Flagship", "West"), (2, "Annex", "West"),
              (3, "Depot", "Central"), (4, "Outlet", "East")]
    products = [(i, f"Prod{i:02d}", CATEGORIES[(i - 1) % 4],
                 round(rng.uniform(20, 400), 2)) for i in range(1, 13)]
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", customers)
    conn.executemany("INSERT INTO stores VALUES (?,?,?)", stores)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    orders, items, item_id = [], [], 1
    for oid in range(1, 46):
        cust = rng.randint(1, 12)
        store = rng.randint(1, 4)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"2024-{month:02d}-{day:02d}"
        ship_lag = rng.randint(1, 20)                 # may cross a quarter boundary
        sm, sd = month, day + ship_lag
        if sd > 28:
            sm, sd = month + 1, sd - 28
        ship_date = f"2024-{min(sm,12):02d}-{sd:02d}"
        status = rng.choices(["completed", "returned", "cancelled"],
                             weights=[80, 12, 8])[0]
        order_total = 0.0
        for _ in range(rng.randint(1, 4)):            # line items
            pid = rng.randint(1, 12)
            qty = rng.randint(1, 5)
            list_price = products[pid - 1][3]
            unit_price = round(list_price * rng.uniform(0.7, 0.98), 2)  # sold below list
            items.append((item_id, oid, pid, qty, unit_price))
            order_total += qty * list_price       # GROSS: list price, not what it sold for
            item_id += 1
        orders.append((oid, cust, store, order_date, ship_date, status,
                       round(order_total, 2)))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    conn.commit()
    return conn


def run_sql(conn: sqlite3.Connection, sql: str):
    """Run one query and return its rows, or raise if it doesn't execute."""
    return conn.execute(sql).fetchall()


# ── The business questions, each with a hand-written gold query ────────────────
# Every gold query is cross-checked against an independent pure-Python computation
# in scripts/_datasql_probe.py, so the "gold" really is gold (the honest-
# measurement lesson: a large fraction of public text-to-SQL gold answers are
# themselves wrong, so verify your own).

QUESTIONS = [
    ("q3_revenue",
     "What was the total revenue in Q3 2024 (July through September)?",
     "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) "
     "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
     "WHERE o.status = 'completed' "
     "AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30';"),
    ("revenue_by_region",
     "What was the total revenue by region in Q3 2024?",
     "SELECT c.region, ROUND(SUM(oi.quantity * oi.unit_price), 2) "
     "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
     "JOIN customers c ON c.customer_id = o.customer_id "
     "WHERE o.status = 'completed' "
     "AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30' "
     "GROUP BY c.region;"),
    ("units_by_category",
     "How many units did each product category sell in Q3 2024?",
     "SELECT p.category, SUM(oi.quantity) AS units "
     "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
     "JOIN products p ON p.product_id = oi.product_id "
     "WHERE o.status = 'completed' "
     "AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30' "
     "GROUP BY p.category;"),
    ("active_customers",
     "How many distinct customers placed a completed order in Q3 2024?",
     "SELECT COUNT(DISTINCT o.customer_id) "
     "FROM orders o WHERE o.status = 'completed' "
     "AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30';"),
    ("avg_order_value",
     "What was the average value of a completed order in Q3 2024?",
     "SELECT ROUND(SUM(oi.quantity * oi.unit_price) / "
     "COUNT(DISTINCT o.order_id), 2) "
     "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
     "WHERE o.status = 'completed' "
     "AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30';"),
]


# ── Letting the model write the SQL ───────────────────────────────────────────

def _schema_prompt(question: str) -> str:
    return ("You are a data analyst. Write ONE SQLite query that answers the "
            "question against this schema. Output only the SQL in a ```sql code "
            "block.\n\n" + SCHEMA_DDL + "\nQuestion: " + question)


def _extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply: prefer a fenced block, else first SELECT."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    sel = re.search(r"(SELECT\b.*?;)", text, re.S | re.I)
    return sel.group(1).strip() if sel else text.strip()


def ask_sql(question: str, model: str = EIS_MODEL) -> str:
    """Ask the model for a query. gpt-oss reasons in a hidden channel and often
    leaves `content` empty, so we scan both channels for the SQL it wrote."""
    msg = ollama.chat(model=model, think=True, options={"num_predict": 1500},
                      messages=[{"role": "user",
                                 "content": _schema_prompt(question)}])["message"]
    text = msg.get("content") or msg.get("thinking") or ""
    return _extract_sql(text)


# ── The checker: correct / silent-wrong / crash ───────────────────────────────

def _canon(rows) -> tuple:
    """Canonical form of a result set: row order and column order made irrelevant,
    numbers rounded to cents, strings folded. Two result sets that mean the same
    thing canonicalize equal; a wrong answer doesn't."""
    out = []
    for row in rows:
        cells = []
        for c in row:
            if isinstance(c, (int, float)):
                cells.append(round(float(c), 2))
            elif c is None:
                cells.append(None)
            else:
                cells.append(str(c).strip().casefold())
        out.append(tuple(sorted(cells, key=lambda v: (v is None, repr(v)))))
    return tuple(sorted(out, key=repr))


def classify(conn: sqlite3.Connection, model_sql: str, gold_sql: str) -> str:
    """Run the model's SQL beside the gold and bin the outcome."""
    gold = _canon(run_sql(conn, gold_sql))
    try:
        got = _canon(run_sql(conn, model_sql))
    except Exception:
        return "crash"
    return "correct" if got == gold else "silent_wrong"


def audit_sql(model: str = EIS_MODEL, trials: int = 2) -> dict:
    """Score the model's free-form SQL over every question, ``trials`` times each."""
    conn = build_warehouse()
    tally = {"correct": 0, "silent_wrong": 0, "crash": 0, "n": 0}
    for _, question, gold in QUESTIONS:
        for _ in range(trials):
            tally[classify(conn, ask_sql(question, model), gold)] += 1
            tally["n"] += 1
    return tally


# ── The fix: a governed semantic layer ────────────────────────────────────────
# Instead of free-form SQL, the model picks a defined metric and an optional
# dimension by name. A deterministic compiler turns the pick into the one correct
# query, with the joins, the status filter, the date column, and the metric
# definition all baked in where the model can't get them wrong. The only mistake
# left is choosing the wrong name, a far smaller and checkable error surface.

METRICS = {
    "revenue":         "SUM(oi.quantity * oi.unit_price)",
    "units_sold":      "SUM(oi.quantity)",
    "active_customers": "COUNT(DISTINCT o.customer_id)",
    "order_count":     "COUNT(DISTINCT o.order_id)",
    "avg_order_value": "SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.order_id)",
}
DIMENSIONS = {
    "region":   "c.region",      # canonical: the customer's billing region
    "category": "p.category",
}
_NEEDS_ITEMS = {"revenue", "units_sold", "avg_order_value"}


def compile_metric(metric: str, by: str = None, period: str = None) -> str:
    """Compile a (metric, dimension, period) pick into the one correct SQL, with
    only the joins that pick actually needs. The metric definition, the status
    filter, and the date window are baked in where the model can't get them wrong."""
    select = METRICS[metric]
    joins = ""
    if metric in _NEEDS_ITEMS:
        joins += " JOIN order_items oi ON oi.order_id = o.order_id"
    if by == "region":
        joins += " JOIN customers c ON c.customer_id = o.customer_id"
    if by == "category":
        joins += " JOIN products p ON p.product_id = oi.product_id"
    where = "WHERE o.status = 'completed'"
    if period == "Q3-2024":
        where += " AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30'"
    if by:
        dim = DIMENSIONS[by]
        return f"SELECT {dim}, ROUND({select}, 2) FROM orders o{joins} {where} GROUP BY {dim};"
    return f"SELECT ROUND({select}, 2) FROM orders o{joins} {where};"


def ask_semantic(question: str, model: str = EIS_MODEL) -> dict:
    """Ask the model to pick a metric/dimension/period from the governed layer."""
    prompt = ("Answer the question by choosing from this governed semantic layer. "
              "Reply with ONLY a JSON object {\"metric\": ..., \"by\": ... or null, "
              "\"period\": \"Q3-2024\" or null}.\n"
              f"metrics: {list(METRICS)}\ndimensions: {list(DIMENSIONS)}\n"
              f"Question: {question}")
    msg = ollama.chat(model=model, think=True, options={"num_predict": 1200},
                      messages=[{"role": "user", "content": prompt}])["message"]
    text = msg.get("content") or msg.get("thinking") or ""
    blob = re.search(r"\{[^{}]*\}", text, re.S)
    try:
        return json.loads(blob.group(0)) if blob else {}
    except json.JSONDecodeError:
        return {}


def classify_semantic(conn, spec: dict, gold_sql: str) -> str:
    """Compile the model's pick and bin it against gold (a bad name is a crash)."""
    gold = _canon(run_sql(conn, gold_sql))
    try:
        sql = compile_metric(spec.get("metric"), spec.get("by"), spec.get("period"))
        got = _canon(run_sql(conn, sql))
    except Exception:
        return "crash"
    return "correct" if got == gold else "silent_wrong"


def audit_semantic(model: str = EIS_MODEL, trials: int = 2) -> dict:
    conn = build_warehouse()
    tally = {"correct": 0, "silent_wrong": 0, "crash": 0, "n": 0}
    for _, question, gold in QUESTIONS:
        for _ in range(trials):
            tally[classify_semantic(conn, ask_semantic(question, model), gold)] += 1
            tally["n"] += 1
    return tally


# ── Baked results (captured by scripts/_datasql_probe.py, gpt-oss:20b, trials=3) ─
# Free-form SQL: 8 of 15 queries ran and silently disagreed with gold, 0 crashed.
# The governed layer: the model picked the right metric every time, 15/15 correct.
# avg_order_value was wrong on all three free-form tries (always AVG of the gross
# order_total); active_customers, the one pure count, was right on all three.
SQL_AUDIT = {
    "raw":      {"correct": 7, "silent_wrong": 8, "crash": 0, "n": 15},
    "semantic": {"correct": 15, "silent_wrong": 0, "crash": 0, "n": 15},
}

# One real hero: the model's SQL ran fine and returned a number that looks right
# and isn't. Captured verbatim; gold_value computed from the warehouse. ``flaw``
# is the one-line human reading for the prose below the box.
COLD_OPEN = {
    "question": "What was the total revenue in Q3 2024 (July through September)?",
    "model_sql": "SELECT SUM(order_total) AS total_revenue\n"
                 "FROM orders\n"
                 "WHERE status = 'completed'\n"
                 "  AND date(order_date) BETWEEN '2024-07-01' AND '2024-09-30';",
    "model_value": 13781.85,
    "gold_value": 11756.71,
    "flaw": "summed order_total, the gross list-price column, not the net "
            "line-item revenue the business means",
}

# One real semantic-layer round trip: the model's metric pick and the correct
# number the compiler produced from it.
SEMANTIC_DEMO = {
    "question": "What was the total revenue in Q3 2024 (July through September)?",
    "spec": {"metric": "revenue", "by": None, "period": "Q3-2024"},
    "compiled_sql": "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) "
                    "FROM orders o "
                    "JOIN order_items oi ON oi.order_id = o.order_id "
                    "WHERE o.status = 'completed' "
                    "AND o.order_date >= '2024-07-01' AND o.order_date <= '2024-09-30';",
    "value": 11756.71,
}


def _pretty_sql(sql: str) -> str:
    """Break a one-line query at its clause keywords so it reads like SQL on the
    page. Same tokens, just wrapped; the query itself is unchanged."""
    s = " ".join(sql.split())
    for kw in (" FROM ", " JOIN ", " WHERE ", " GROUP BY ", " ORDER BY ", " LIMIT "):
        s = s.replace(kw, "\n" + kw.strip() + " ")
    return s


def show_cold_open(demo=COLD_OPEN):
    """The model writes SQL, the database runs it, and the number is wrong."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["question"])
    show_code("gpt-oss", _pretty_sql(demo["model_sql"]))
    show_turn("DB", f"-> {demo['model_value']}   (ran clean, no error)")
    show_turn("CHECK", f"gold = {demo['gold_value']}   (the real answer; the "
                       f"model's number is wrong)")


def show_semantic_fix(demo=SEMANTIC_DEMO):
    """The model picks a defined metric; the layer compiles the one right query."""
    from genai.agent import show_code, show_turn
    show_turn("you", demo["question"])
    show_turn("gpt-oss", f"pick = {json.dumps(demo['spec'])}")
    show_code("layer", _pretty_sql(demo["compiled_sql"]))
    show_turn("DB", f"-> {demo['value']}   (the defined metric, computed correctly)")
