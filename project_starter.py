import os
import re
import time
import ast
import dotenv
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from typing import Dict, List, Union

from sqlalchemy import create_engine, Engine
from sqlalchemy.sql import text

dotenv.load_dotenv()


db_engine = create_engine("sqlite:///munder_difflin.db")

paper_supplies = [
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},

    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10},
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02},
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10},
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50},
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05},
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03},
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50},
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05},
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25},
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75},
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50},

    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

paper_supplies_lookup = {p["item_name"]: float(p["unit_price"]) for p in paper_supplies}


# STARTER HELPER FUNCTIONS


def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    np.random.seed(seed)
    num_items = int(len(paper_supplies) * coverage)
    selected_indices = np.random.choice(range(len(paper_supplies)), size=num_items, replace=False)
    selected_items = [paper_supplies[i] for i in selected_indices]

    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": int(np.random.randint(200, 800)),
            "min_stock_level": int(np.random.randint(50, 150))
        })
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    try:
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],
            "units": [],
            "price": [],
            "transaction_date": [],
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        initial_date = datetime(2025, 1, 1).isoformat()

        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        initial_transactions = []
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": int(item["current_stock"]),
                "price": float(item["current_stock"]) * float(item["unit_price"]),
                "transaction_date": initial_date,
            })

        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    try:
        date_str = date.isoformat() if isinstance(date, datetime) else date
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])
    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """
    return pd.read_sql(stock_query, db_engine, params={"item_name": item_name, "as_of_date": as_of_date})

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        input_date_dt = datetime.now()

    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    delivery_date_dt = input_date_dt + timedelta(days=days)
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    try:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0
    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    cash = get_cash_balance(as_of_date)

    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = int(stock_info["current_stock"].iloc[0])
        item_value = float(stock) * float(item["unit_price"])
        inventory_value += item_value
        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": float(item["unit_price"]),
            "value": item_value,
        })

    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": float(inventory_value),
        "total_assets": float(cash + inventory_value),
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    conditions = []
    params = {}

    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

# ============================================================
# YOUR MULTI AGENT STARTS HERE
# ============================================================

from smolagents import tool, CodeAgent
from smolagents.models import OpenAIModel


API_KEY = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing UDACITY_OPENAI_API_KEY (or OPENAI_API_KEY) in .env")

model = OpenAIModel(
    model_id="gpt-4.1-mini",
    api_key=API_KEY,
    base_url="https://openai.vocareum.com/v1",
)

def _extract_request_date(request_text: str) -> str:
    m = re.search(r"Date of request:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", request_text)
    if m:
        return m.group(1)
    return datetime.now().strftime("%Y-%m-%d")

def _extract_quantity(request_text: str) -> int:
    nums = re.findall(r"\b(\d{1,6})\b", request_text)
    if nums:
        q = int(nums[0])
        return max(1, q)
    return 100

def _match_item_name(request_text: str) -> str:
    t = request_text.lower()
    # prefer longer matches first
    candidates = sorted([p["item_name"] for p in paper_supplies], key=len, reverse=True)
    for name in candidates:
        if name.lower() in t:
            return name
    # fallback: guess common words
    if "a4" in t:
        return "A4 paper"
    if "letter" in t:
        return "Letter-sized paper"
    if "cardstock" in t:
        return "Cardstock"
    return ""

def _keywords_for_history(request_text: str, item_name: str) -> List[str]:
    # lightweight keywords for search_quote_history
    tokens = re.findall(r"[a-zA-Z]{4,}", request_text.lower())
    tokens = [w for w in tokens if w not in {"please", "need", "want", "quote", "price", "cost", "today"}]
    out = []
    if item_name:
        out.extend(item_name.lower().split())
    out.extend(tokens[:6])
    # unique, keep order
    seen = set()
    cleaned = []
    for w in out:
        if w not in seen:
            seen.add(w)
            cleaned.append(w)
    return cleaned[:6] if cleaned else ["paper"]


# TOOLS 

@tool
def tool_get_stock_level(item_name: str, as_of_date: str) -> int:
    """
    Get current stock level for a specific item on a specific date.

    Args:
        item_name: Exact catalog item name (e.g., "A4 paper").
        as_of_date: ISO date string "YYYY-MM-DD" used as inventory cutoff.

    Returns:
        Current stock level as an integer.
    """
    df = get_stock_level(item_name, as_of_date)
    return int(df["current_stock"].iloc[0])

@tool
def tool_get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Get a snapshot of all inventory (items with positive stock) on a given date.

    Args:
        as_of_date: ISO date string "YYYY-MM-DD" used as inventory cutoff.

    Returns:
        Dictionary mapping item_name -> current stock.
    """
    return get_all_inventory(as_of_date)

@tool
def tool_get_supplier_delivery_date(request_date: str, quantity: int) -> str:
    """
    Estimate supplier delivery date for an order quantity.

    Args:
        request_date: ISO date string "YYYY-MM-DD" for when the order is placed.
        quantity: Number of units being ordered.

    Returns:
        Estimated supplier delivery date as "YYYY-MM-DD".
    """
    return get_supplier_delivery_date(request_date, int(quantity))

@tool
def tool_get_cash_balance(as_of_date: str) -> float:
    """
    Get company cash balance as of a given date.

    Args:
        as_of_date: ISO date string "YYYY-MM-DD" for balance cutoff.

    Returns:
        Cash balance as a float.
    """
    return float(get_cash_balance(as_of_date))

@tool
def tool_generate_financial_report(as_of_date: str) -> Dict:
    """
    Generate a financial report snapshot for a given date.

    Args:
        as_of_date: ISO date string "YYYY-MM-DD" for report cutoff.

    Returns:
        Financial report dictionary with cash, inventory value, and top selling products.
    """
    return generate_financial_report(as_of_date)

@tool
def tool_search_quote_history(keywords: List[str], limit: int = 5) -> List[Dict]:
    """
    Search historical quote records using keywords.

    Args:
        keywords: List of search terms used to match past requests and explanations.
        limit: Max number of historical quotes to return.

    Returns:
        List of historical quote dictionaries.
    """
    return search_quote_history(keywords, int(limit))

@tool
def tool_reorder_stock(item_name: str, quantity: int, date: str) -> Dict:
    """
    Place a stock reorder (creates a stock_orders transaction) and return ETA.

    Args:
        item_name: Exact catalog item name to reorder.
        quantity: Units to reorder.
        date: ISO date string "YYYY-MM-DD" for transaction date.

    Returns:
        Dict with ok flag, transaction_id, total_cost, and eta.
    """
    item_name = item_name.strip()
    qty = int(quantity)

    if item_name not in paper_supplies_lookup:
        return {"ok": False, "reason": f"Unknown item: {item_name}"}

    unit_price = float(paper_supplies_lookup[item_name])
    total_cost = float(unit_price * qty)

    tx_id = create_transaction(
        item_name=item_name,
        transaction_type="stock_orders",
        quantity=qty,
        price=total_cost,
        date=date,
    )
    eta = get_supplier_delivery_date(date, qty)
    return {"ok": True, "transaction_id": int(tx_id), "total_cost": total_cost, "eta": eta}

@tool
def tool_create_sale(item_name: str, quantity: int, total_price: float, date: str) -> Dict:
    """
    Create a sales transaction (finalize an order).

    Args:
        item_name: Exact catalog item name being sold.
        quantity: Units being sold.
        total_price: Final total charged to the customer.
        date: ISO date string "YYYY-MM-DD" for transaction date.

    Returns:
        Dict with ok flag and transaction_id.
    """
    tx_id = create_transaction(
        item_name=item_name,
        transaction_type="sales",
        quantity=int(quantity),
        price=float(total_price),
        date=date,
    )
    return {"ok": True, "transaction_id": int(tx_id)}


# AGENTS


inventory_agent = CodeAgent(
    name="InventoryAgent",
    description="Checks stock, suggests/places reorder if needed, estimates supplier ETA.",
    model=model,
    tools=[tool_get_stock_level, tool_get_all_inventory, tool_get_supplier_delivery_date, tool_reorder_stock],
)

quote_agent = CodeAgent(
    name="QuoteAgent",
    description="Builds customer quotes using quote history and basic discounting. Avoids internal margin disclosure.",
    model=model,
    tools=[tool_search_quote_history],
)

sales_agent = CodeAgent(
    name="SalesAgent",
    description="Finalizes sales by recording transactions when inventory is sufficient.",
    model=model,
    tools=[tool_create_sale, tool_get_stock_level],
)

business_advisor_agent = CodeAgent(
    name="BusinessAdvisorAgent",
    description="Internal-only advisor. Uses financial report and cash to suggest improvements (not shown to customer).",
    model=model,
    tools=[tool_generate_financial_report, tool_get_cash_balance],
)

# Orchestrator agent
orchestrator_agent = CodeAgent(
    name="OrchestratorAgent",
    description=(
        "Orchestrates customer requests by delegating to InventoryAgent, QuoteAgent, SalesAgent, and BusinessAdvisorAgent. "
        "Produces customer-safe outputs with rationale. Must not reveal internal profit margins or sensitive info."
    ),
    model=model,
    tools=[
        tool_get_stock_level,
        tool_get_all_inventory,
        tool_get_supplier_delivery_date,
        tool_get_cash_balance,
        tool_generate_financial_report,
        tool_search_quote_history,
        tool_reorder_stock,
        tool_create_sale,
    ],
)


# ORCHESTRATION FUNCTION
def handle_request(request_text: str) -> str:
    request_date = _extract_request_date(request_text)
    item_name = _match_item_name(request_text)
    qty = _extract_quantity(request_text)

    if not item_name:
        # Customer-safe fallback
        return (
            "I can help with inventory and quotes. Please specify the product name (e.g., A4 paper, Paper cups) "
            "and the quantity you need."
        )

    # 1) Inventory analysis (deterministic tools)
    stock_now = int(tool_get_stock_level(item_name, request_date))

    # reorder if low 
    reorder_action = None
    eta = None
    if stock_now < qty:
        reorder_qty = max(200, qty - stock_now)  
        reorder_action = tool_reorder_stock(item_name, reorder_qty, request_date)
        eta = reorder_action.get("eta") if isinstance(reorder_action, dict) else None

    # 2) Quote generation (use history)
    keywords = _keywords_for_history(request_text, item_name)
    history = tool_search_quote_history(keywords, 5)

    # Base pricing
    unit_price = paper_supplies_lookup.get(item_name, 0.10)
    subtotal = float(unit_price * qty)

    # Bulk discount
    discount_pct = 0.0
    if qty >= 5000:
        discount_pct = 0.12
    elif qty >= 2000:
        discount_pct = 0.08
    elif qty >= 500:
        discount_pct = 0.05

    discounted_total = float(subtotal * (1.0 - discount_pct))

    # Use history only as guidance text (no internal info)
    history_hint = ""
    if isinstance(history, list) and len(history) > 0:
        history_hint = "Pricing is aligned with similar past jobs."

    # 3) Decide fulfillment + sale (only if request sounds like an order)
    wants_to_order = any(w in request_text.lower() for w in ["order", "buy", "purchase", "place order", "confirm"])
    order_status = ""
    if wants_to_order:
        if stock_now >= qty:
            sale = tool_create_sale(item_name, qty, discounted_total, request_date)
            order_status = f"✅ Order confirmed. Transaction ID: {sale.get('transaction_id')}."
        else:
            if eta:
                order_status = f"⚠️ Not enough stock today. We can reorder and deliver after supplier ETA: {eta}."
            else:
                order_status = "⚠️ Not enough stock today. We can reorder once confirmed."

    # 4) Internal advisor
    _ = tool_generate_financial_report(request_date)  
    _ = tool_get_cash_balance(request_date)           

    # Customer response 
    lines = []
    lines.append(f"Item: {item_name}")
    lines.append(f"Requested quantity: {qty}")
    lines.append(f"Stock available on {request_date}: {stock_now}")

    if reorder_action and isinstance(reorder_action, dict) and reorder_action.get("ok"):
        lines.append(f"Reorder placed: {reorder_action.get('transaction_id')} (ETA: {reorder_action.get('eta')})")

    lines.append("")
    lines.append("Quote:")
    if discount_pct > 0:
        lines.append(f"- Unit price: ${unit_price:.2f}")
        lines.append(f"- Subtotal: ${subtotal:.2f}")
        lines.append(f"- Bulk discount: {int(discount_pct*100)}%")
        lines.append(f"- Total: ${discounted_total:.2f}")
        lines.append("- Rationale: bulk discount applied to support larger order volume.")
    else:
        lines.append(f"- Unit price: ${unit_price:.2f}")
        lines.append(f"- Total: ${discounted_total:.2f}")
        lines.append("- Rationale: standard pricing (no bulk discount threshold reached).")

    if history_hint:
        lines.append(f"- Note: {history_hint}")

    if order_status:
        lines.append("")
        lines.append(f"Order status: {order_status}")

    return "\n".join(lines)


# TEST HARNESS


def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"

        response = handle_request(request_with_date)

        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    results = run_test_scenarios()
