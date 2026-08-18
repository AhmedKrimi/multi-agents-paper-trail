# Beaver's Choice Paper Company — Multi-Agent Inventory & Quoting System

<p align="center">
  <img src="images/beavers_choice.jpeg" alt="Beaver's Choice Paper Company" width="300"/>
</p>

An end-to-end multi-agent system that automates the complete order lifecycle of a paper supply company — from a free-text customer email to a booked sale in the ledger. Built with [`smolagents`](https://github.com/huggingface/smolagents), Pydantic, and SQLite.

## The Challenge

The Beaver's Choice Paper Company was losing sales to slow, error-prone manual operations. Every incoming customer request required someone to:

- **Interpret the request** — customers write in free text ("we need glossy sheets for a gala next Friday"), often naming products loosely and mixing several items in one message
- **Check inventory in real time** — stock levels change with every sale, and a quote based on stale numbers means a broken promise
- **Decide on restocking** — when stock is short, is it worth ordering from the supplier? Can it arrive before the customer's deadline? Is there enough cash?
- **Price competitively** — discounts should reflect order size and past quoting behavior, not gut feeling
- **Close the sale reliably** — record the transaction without double-charging, overselling, or corrupting the ledger

Doing this by hand doesn't scale; automating it naively with a single LLM prompt is unreliable — one hallucinated stock number or invented delivery date corrupts real financial data.

## The Solution

This project solves the problem with a **pipeline of five specialized agents**, each owning one stage of the order lifecycle and communicating through a strictly validated data contract:

- A single **Pydantic model (`RequestedItem`)** flows through the pipeline; each agent enriches it, so every decision is machine-checkable and traceable per item.
- **All financial and inventory logic lives in deterministic Python tools** — the LLM decides *when* to act, never *what the numbers are*. Stock math, discount rules, supplier reorders, and ledger writes are pure code.
- **Defensive guards at every stage** — enforced execution order, duplicate-sale protection, shortfall re-verification at sale time, and cash-balance checks before any supplier purchase — keep a single agent mistake from cascading into the books.

The result: a customer request goes in as plain text and comes out as a priced, verified, executed (or transparently declined) order — with the database as the single source of truth at every point in time.

> Built as the capstone for Udacity's Agentic AI program (multi-agent systems module).

---

## Architecture

The system uses exactly **5 agents**, all implemented as `ToolCallingAgent` instances and coordinated by a top-level orchestrator:

```
                        ┌──────────────────┐
 Customer request ────▶ │   Orchestrator    │
                        └────────┬─────────┘
          ┌──────────────┬───────┴──────┬───────────────┐
          ▼              ▼              ▼               ▼
 ┌────────────────┐ ┌─────────────┐ ┌────────────┐ ┌──────────────┐
 │ Order Processor│ │  Inventory  │ │   Quote    │ │    Sales     │
 │    Agent       │ │   Manager   │ │  Manager   │ │   Manager    │
 └────────────────┘ └─────────────┘ └────────────┘ └──────────────┘
   assign_item       assign_inventory  calculate_    execute_sale
                     _name              discount     (+ reorder_supply)
                     check_inventory
                     check_delivery_
                     timeline
```

| Agent | Responsibility | Tools |
|---|---|---|
| **Orchestrator** | Coordinates the whole workflow and composes the final customer-facing answer | `get_order_details`, `manage_inventory`, `prepare_quote`, `prepare_sale` |
| **Order Processor** | Extracts every requested item from the free-text customer message and wraps it in a structured `RequestedItem` object with a unique suborder ID | `assign_item` |
| **Inventory Manager** | Maps customer wording to a valid catalog name, checks stock levels as of the order date, and verifies whether missing quantities can arrive from the supplier before the requested delivery date | `assign_inventory_name`, `check_inventory`, `check_delivery_timeline` |
| **Quote Manager** | Applies a discount based on historical quote data (order size from `search_quote_history`) or, when no history exists, on requested quantity | `calculate_discount` |
| **Sales Manager** | Reorders shortfalls from the supplier (if cash and timing allow), re-verifies stock, and records the sale transaction | `execute_sale` |

### Data flow

Every item travels through the pipeline as a **Pydantic `RequestedItem`** model (see `items_model.py`), which acts as the shared contract between agents. Each stage enriches the object:

1. `suborder_id`, `item_info`, `requested_quantity`, `order_date`, `delivery_date` (Order Processor)
2. `inventory_name`, `can_fulfil`, `short_by`, `order_supplier`, `estimated_date` (Inventory Manager)
3. `discount` (Quote Manager)
4. `suborder_executed` (Sales Manager)

The orchestrator enforces ordering: each downstream tool refuses to run if the previous stage's result is missing (e.g. `prepare_quote` returns an error message if `manage_inventory` hasn't been called yet).

---

## Business logic

### Inventory & reordering
- Stock levels are computed from the `transactions` table (stock orders minus sales) as of the request date via `get_stock_level`.
- If stock is insufficient, `check_delivery_timeline` uses `get_supplier_delivery_date` (lead time scales with quantity: same-day up to 10 units, up to 7 days for >1000 units) to decide whether a supplier reorder can arrive before the customer's delivery date.
- `reorder_supply` purchases the shortfall at **75% of the retail unit price** (`SUPPLIER_PRICE_FACTOR = 0.75`), but only if the current cash balance covers the cost. Purchases are logged as `stock_orders` transactions.

### Quoting & discounts
`calculate_discount` first searches historical quotes (`search_quote_history`) for a matching request:

- History found → discount by past `order_size`: **large = 20%**, **medium = 10%**, otherwise 0%.
- No history → discount by requested quantity: **≥1000 units = 20%**, **≥100 units = 10%**, else 0%.

### Sales execution
`execute_sale` guards against several failure modes:
- **Duplicate execution** — a module-level `_items_sold` set prevents double-selling the same suborder ID within one order.
- **Stale shortfall data** — the actual shortfall is recomputed at sale time and compared against the Inventory Manager's figure; on mismatch the sale is aborted.
- **Final stock check** — stock is re-verified after any supplier reorder before the sale transaction is written.
- Successful sales are recorded at `(1 − discount) × unit_price × quantity` as `sales` transactions.

The orchestrator's final answer reports one of three verdicts: **fulfilled**, **partially fulfilled** (listing which items shipped and which didn't, with delivery dates), or **cannot be fulfilled**. Its prompt explicitly forbids fabricating transactions, dates, or orders.

---

## Project structure

```
.
├── main.py                    # Entry point: builds Context, runs the test-scenario loop
├── agents.py                  # The five agents + Orchestrator, each owning its tools
├── config.py                  # Static values: db engine, catalog names, unit prices, working dir
├── context.py                 # Context dataclass — shared state injected into agents/tools
├── utils.py                   # DB helpers, date parsing, reorder_supply, validate_item
├── inventory.py               # PAPER_SUPPLIES catalog (names, categories, unit prices)
├── items_model.py             # RequestedItem Pydantic model (shared agent contract)
├── quote_requests.csv         # Historical customer inquiries (seeds quote_requests table)
├── quotes.csv                 # Historical quotes (seeds quotes table)
├── quote_requests_sample.csv  # Test scenarios
├── munder_difflin.db          # SQLite database (created at runtime)
└── .env                       # OPENAI_API_KEY (not committed)
```

### Database (SQLite via SQLAlchemy)

| Table | Purpose |
|---|---|
| `transactions` | Ledger of all `stock_orders` and `sales` (drives stock levels and cash balance) |
| `inventory` | Reference table of stocked items (~40% random coverage of the catalog, seeded) |
| `quote_requests` | Historical customer inquiries |
| `quotes` | Historical quotes with job type, order size, and event type metadata |

The company starts with a **$50,000 cash balance** and randomized initial stock.

---

## Setup & usage

### Requirements

- Python 3.10+
- An OpenAI API key (the system uses `gpt-4o-mini` via smolagents' `OpenAIServerModel`)

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the parent directory of the source folder:

```
OPENAI_API_KEY=sk-...
```

The model is instantiated with `temperature=0.0` and `parallel_tool_calls=False` for deterministic, sequential tool execution.

### Run

```bash
python main.py
```

This will:
1. Initialize the database (`init_database`) — tables, historical quotes, seeded inventory, starting cash.
2. Load and date-sort the test scenarios from `quote_requests_sample.csv`.
3. Process each request through the orchestrator, printing the response plus updated cash and inventory value after every order.
4. Print a final financial report and write all results to `test_results.csv`.

---

## Design decisions

- **Structured state over free text.** Passing a validated Pydantic model between agents (serialized as JSON) keeps every stage's output machine-checkable and prevents the LLM from silently dropping fields.
- **Deterministic tools, LLM for orchestration only.** All money- and inventory-affecting logic (stock math, discount rules, reorder decisions, transaction writes) lives in plain Python tools. The LLM decides *when* to call them, not *what the numbers are*.
- **Guard rails at every stage.** Stage-ordering checks in the orchestrator's tools, duplicate-sale protection, shortfall re-verification, and cash-balance checks before supplier orders all limit the blast radius of any single agent mistake.
- **Point-in-time correctness.** Stock and cash are always evaluated *as of the request date*, so replaying historical requests in order produces a consistent ledger.

## Possible improvements (WIP)

- Catalog name matching relies on the LLM choosing the closest valid name; a fuzzy-matching fallback could make it more robust.
- Discounts use only the top-1 historical match; aggregating over several similar quotes could yield better pricing.
- No proactive restocking to `min_stock_level` — reorders happen only when a specific order is short.