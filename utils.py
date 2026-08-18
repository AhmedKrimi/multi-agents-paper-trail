import ast
import os
from datetime import datetime, timedelta
from config import working_dir
from items_model import RequestedItem

import numpy as np
import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.sql import text
from context import Context

from inventory import PAPER_SUPPLIES


def _parse_date(value) -> datetime:
    """Coerce anything date-like to a datetime object."""
    return datetime.fromisoformat(str(value).split("T")[0])


def _as_date(value) -> str:
    """Coerce anything date-like to YYYY-MM-DD."""
    return _parse_date(value).strftime("%Y-%m-%d")


# Given below are some utility functions you can use to implement your multi-agent system
def generate_sample_inventory(
    PAPER_SUPPLIES: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `PAPER_SUPPLIES` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        PAPER_SUPPLIES (list): A list of dictionaries, each representing a paper item with
                               keys 'inventory_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - inventory_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(PAPER_SUPPLIES) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(PAPER_SUPPLIES)), size=num_items, replace=False
    )

    # Extract selected items from PAPER_SUPPLIES list
    selected_items = [PAPER_SUPPLIES[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "inventory_name": item["inventory_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                # Realistic stock range
                "current_stock": np.random.randint(200, 800),
                # Reasonable threshold for reordering
                "min_stock_level": np.random.randint(50, 150),
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def get_all_inventory(db_engine: Engine, as_of_date: str) -> dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        db_engine (Engine) : database engine
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            inventory_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE inventory_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY inventory_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {inventory_name: stock}
    return dict(zip(result["inventory_name"], result["stock"]))


def get_stock_level(
    db_engine: Engine, inventory_name: str, as_of_date: str | datetime
) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        db_engine (Engine): Database engine
        inventory_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'inventory_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = _as_date(as_of_date.isoformat())

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            inventory_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE inventory_name = :inventory_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"inventory_name": inventory_name, "as_of_date": as_of_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'"
    )

    # Attempt to parse the input date
    try:
        input_date_dt = _parse_date(input_date_str)
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base."
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return _as_date(delivery_date_dt)


def get_cash_balance(db_engine: Engine, as_of_date: str | datetime) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        db_engine (Engine): Database Engine
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = _as_date(as_of_date)

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def reorder_supply(
    inventory_name: str, quantity: int, order_date: str, ctx: Context
) -> bool:
    """Order item from the supplier
    Args:
        inventory_name (str): The name of the item to reorder from the supplier.
        quantity (int): Number of units to reorder from the supplier.
        order_date (str): Date of the order in ISO 8601 format.

    Returns:
        bool: True, if Transaction is OK, otherwise False
    """
    if quantity <= 0:
        print(
            f"no ordering is needed for {inventory_name}, quantity: {quantity}")
        return False

    SUPPLIER_PRICE_FACTOR = 0.75

    if inventory_name not in ctx.available_items:
        print(
            f"Ordering transaction from the supplier of {inventory_name} has failed, {inventory_name} is not valid item"
        )
        return False

    unit_price = ctx.products_price[inventory_name]
    price = round(SUPPLIER_PRICE_FACTOR * unit_price * quantity, 2)

    try:
        if price <= get_cash_balance(ctx.db_engine, order_date):
            res = create_transaction(
                ctx.db_engine,
                inventory_name,
                transaction_type="stock_orders",
                quantity=quantity,
                price=price,
                date=order_date,
            )
            print(
                f"Ordering {inventory_name} transaction was successful - transaction ID: {res}"
            )
            return True
        else:
            print(
                f"Not enough cash to reorder {inventory_name} transaction has failed")
            return False
    except Exception as e:
        print(f"Ordering {inventory_name} transaction has failed : {e}")
        return False


def generate_financial_report(db_engine: Engine, as_of_date: str | datetime) -> dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        db_engine (Engine): Database engine
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = _as_date(as_of_date)

    # Get current cash balance
    cash = get_cash_balance(db_engine, as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(
            db_engine, item["inventory_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append(
            {
                "inventory_name": item["inventory_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT inventory_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY inventory_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine,
                            params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(
    db_engine: Engine, search_terms: list[str], limit: int = 5
) -> list[dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        db_engine (Engine): Database engine
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " OR ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
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

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "inventory_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = _as_date(str(datetime(2025, 1, 1)))

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(
            os.path.join(working_dir, "quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(os.path.join(working_dir, "quotes.csv"))
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(PAPER_SUPPLIES, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "inventory_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "inventory_name": item["inventory_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine,
                            if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(
    db_engine: Engine, inventory_name, transaction_type, quantity, price, date
) -> int:
    try:
        date_str = _as_date(date)

        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError(
                "Transaction type must be 'stock_orders' or 'sales'")

        with db_engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO transactions
                        (inventory_name, transaction_type, units, price, transaction_date)
                        VALUES (:inventory_name, :ttype, :units, :price, :tdate)"""),
                {
                    "inventory_name": inventory_name,
                    "ttype": transaction_type,
                    "units": quantity,
                    "price": price,
                    "tdate": date_str,
                },
            )
            row_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()

            return int(row_id)

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def validate_item(item_dict: dict) -> RequestedItem:
    """Build a RequestedItem, raises ValidationError on bad input"""
    return RequestedItem(**item_dict)
