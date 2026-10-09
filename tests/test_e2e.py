import os

import dotenv
import pandas as pd
import pytest
from smolagents import OpenAIServerModel
from supply_chain_agents.agents import Orchestrator
from supply_chain_agents.config import (
    DATA_DIR,
    PROJECT_ROOT,
    catalog_items,
    test_db_engine,
    product_prices,
)
from supply_chain_agents.context import Context
from supply_chain_agents.utils import embed, get_min_stock_levels, init_database


@pytest.fixture(scope="session")
def orchestrator():
    # Load environment variables for the API key
    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the model with the API key
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.0,
        parallel_tool_calls=False,
    )
    # Initializing the context
    context = Context(
        db_engine=test_db_engine,
        catalog_items=catalog_items,
        product_prices=product_prices,
        catalog_embeddings=embed(catalog_items),
    )
    # Initializing the test database
    init_database(db_engine=context.db_engine)
    context.min_stock_levels = get_min_stock_levels(db_engine=context.db_engine)
    orchestrator = Orchestrator(model, context)
    return orchestrator


@pytest.fixture(scope="session")
def requests():
    quote_requests_sample = pd.read_csv(
        os.path.join(DATA_DIR, "quote_requests_sample.csv")
    )
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
    )
    quote_requests_sample.dropna(subset=["request_date"], inplace=True)
    quote_requests_sample = quote_requests_sample.sort_values("request_date")
    requests = [request for _, request in quote_requests_sample.iterrows()]
    return requests


@pytest.fixture(autouse=True)
def reset_db(orchestrator):
    init_database(db_engine=orchestrator.ctx.db_engine)


def prepare_request(requests, idx):
    row = requests[idx]
    request_date = row["request_date"].strftime("%Y-%m-%d")
    request_with_date = f"{row['request']} (Date of request: {request_date})"
    return request_with_date


def get_item(requested_items, item_description):
    item = next(
        item for item in requested_items if item.item_description == item_description
    )
    return item


pytestmark = pytest.mark.e2e


def test_nominal_order_fulfills_from_available_inventory(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=0)

    orchestrator.process_customer_order(request_with_date)

    item = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A4 glossy paper",
    )

    assert item.matched_catalog_item == "A4 glossy paper"
    assert item.requested_quantity == 200
    assert item.can_fulfill is True
    assert item.shortage_quantity == 0
    assert item.sale_completed is True
    assert isinstance(item.order_line_id, str)
    assert orchestrator.ctx.financial_report is not None


def test_mixed_order_handles_catalog_rejection_and_supplier_fulfillment(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=1)
    orchestrator.process_customer_order(request_with_date)
    balloons = get_item(orchestrator.ctx.requested_items, "balloons")

    assert balloons.matched_catalog_item is None
    assert balloons.requested_quantity == 200
    assert balloons.can_fulfill is False
    assert balloons.shortage_quantity == 0
    assert balloons.sale_completed is False
    assert isinstance(balloons.order_line_id, str)

    streamers = get_item(orchestrator.ctx.requested_items, "streamers")

    assert streamers.matched_catalog_item == "Party streamers"
    assert streamers.supplier_order_feasible is True
    assert streamers.supplier_delivery_date is not None
    assert streamers.can_fulfill is True
    assert streamers.sale_completed is True

    assert orchestrator.ctx.financial_report is not None


def test_inventory_shortage_is_replenished_and_sale_completes(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=2)
    orchestrator.process_customer_order(request_with_date)
    item = get_item(
        requested_items=orchestrator.ctx.requested_items, item_description="A3 paper"
    )
    assert item.matched_catalog_item == "A3 paper"
    assert item.requested_quantity == 5000
    assert item.can_fulfill is True
    assert item.supplier_order_feasible is True
    assert item.supplier_delivery_date is not None
    assert item.shortage_quantity == 0
    assert item.sale_completed is True
    assert isinstance(item.order_line_id, str)
    assert orchestrator.ctx.financial_report is not None


def test_catalog_matching_rejects_underspecified_request(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=3)
    orchestrator.process_customer_order(request_with_date)

    item = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="high-quality, recycled cardstock in various colors",
    )
    assert item.matched_catalog_item is None
    assert item.can_fulfill is False
    assert item.sale_completed is False
    assert isinstance(item.order_line_id, str)
    assert orchestrator.ctx.financial_report is not None


def test_semantic_matching_fulfills_non_exact_product_description(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=4)
    orchestrator.process_customer_order(request_with_date)

    item = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="decorative washi tape",
    )
    assert item.matched_catalog_item == "Decorative adhesive tape (washi tape)"
    assert item.requested_quantity == 200
    assert item.can_fulfill is True
    assert item.shortage_quantity == 0
    assert item.sale_completed is True
    assert isinstance(item.order_line_id, str)
    assert orchestrator.ctx.financial_report is not None


def test_semantic_matching_resolves_product_variant_and_fulfills_order(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=5)
    orchestrator.process_customer_order(request_with_date)

    item = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="colorful construction paper",
    )
    assert item.matched_catalog_item == "Construction paper"
    assert item.requested_quantity == 500
    assert item.can_fulfill is True
    assert item.shortage_quantity == 0
    assert item.sale_completed is True
    assert isinstance(item.order_line_id, str)
    assert orchestrator.ctx.financial_report is not None


def test_dimension_based_matching_fulfills_large_format_request(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=6)
    orchestrator.process_customer_order(request_with_date)
    item = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description='poster boards (24" x 36")',
    )

    assert item.matched_catalog_item == "Large poster paper (24x36 inches)"
    assert item.requested_quantity == 300
    assert item.can_fulfill is True
    assert item.shortage_quantity == 0
    assert item.sale_completed is True
    assert isinstance(item.order_line_id, str)
    assert orchestrator.ctx.financial_report is not None


def test_catalog_matching_resolves_size_and_finish_variants(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=7)
    orchestrator.process_customer_order(request_with_date)
    matte = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A4 matte paper",
    )
    recycled = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A4 recycled paper",
    )

    assert matte.matched_catalog_item == "A4 matte paper"
    assert recycled.matched_catalog_item == "A4 recycled paper"


def test_multi_item_matching_keeps_distinct_catalog_variants(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=8)
    orchestrator.process_customer_order(request_with_date)
    a4_paper = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A4 white printer paper",
    )

    a3_glossy = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A3 glossy paper",
    )

    assert a4_paper.matched_catalog_item == "A4 paper"
    assert a4_paper.requested_quantity == 200
    assert a4_paper.can_fulfill is True
    assert a4_paper.shortage_quantity == 0
    assert a4_paper.sale_completed is True
    assert isinstance(a4_paper.order_line_id, str)

    assert a3_glossy.matched_catalog_item == "A3 glossy paper"

    assert orchestrator.ctx.financial_report is not None


def test_supplier_delivery_infeasibility_prevents_sale(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=9)
    orchestrator.process_customer_order(request_with_date)

    a4_paper = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A4 printing paper",
    )

    assert a4_paper.matched_catalog_item == "A4 paper"
    assert a4_paper.can_fulfill is False
    assert a4_paper.shortage_quantity > 0
    assert a4_paper.supplier_order_feasible is False
    assert a4_paper.sale_completed is False


def test_unmatched_order_lines_are_rejected_without_aborting_workflow(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=11)
    orchestrator.process_customer_order(request_with_date)

    glossy = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="high-quality glossy paper",
    )

    cardstock = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="sturdy cardstock",
    )

    assert glossy.matched_catalog_item is None
    assert glossy.sale_completed is False

    assert cardstock.matched_catalog_item is None
    assert cardstock.sale_completed is False

    assert orchestrator.ctx.financial_report is not None


def test_inventory_shortage_and_infeasible_restock_reject_order_lines(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=16)
    orchestrator.process_customer_order(request_with_date)

    failed_descriptions = [
        "A4 white printer paper",
        "A3 colored paper (assorted colors)",
        "table napkins (white)",
        "paper cups (biodegradable)",
    ]

    for description in failed_descriptions:
        item = get_item(orchestrator.ctx.requested_items, description)
        assert item.can_fulfill is False
        assert item.supplier_order_feasible is False
        assert item.sale_completed is False


def test_rejection_due_to_insufficient_cash_despite_stock_reorder_feasibility(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    request_with_date = prepare_request(requests, idx=18)
    orchestrator.process_customer_order(request_with_date)

    a4_paper = get_item(
        requested_items=orchestrator.ctx.requested_items,
        item_description="A4 glossy paper",
    )

    assert a4_paper.matched_catalog_item == "A4 glossy paper"
    assert a4_paper.can_fulfill is False
    assert a4_paper.shortage_quantity > 0
    assert a4_paper.supplier_order_feasible is True
    assert a4_paper.sale_completed is False


def test_new_order_resets_transient_workflow_state(
    orchestrator: Orchestrator, requests: list[pd.Series]
):
    first_request = prepare_request(requests, idx=0)
    orchestrator.process_customer_order(first_request)

    first_items = {item.item_description for item in orchestrator.ctx.requested_items}

    second_request = prepare_request(requests, idx=1)
    orchestrator.process_customer_order(second_request)

    second_items = {item.item_description for item in orchestrator.ctx.requested_items}

    assert first_items.isdisjoint(second_items)
