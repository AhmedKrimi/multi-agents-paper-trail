import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from supply_chain_agents import order_services
from supply_chain_agents.config import (
    catalog_items,
    product_prices,
    test_db_engine,
)
from supply_chain_agents.embeddings import embed
from supply_chain_agents.models import RequestedItem
from supply_chain_agents.order_services import (
    add_order_item,
    assign_inventory_name,
    check_delivery_timeline,
    check_stock_inventory,
    execute_restock,
    execute_sale,
    get_financial_status,
)
from supply_chain_agents.utils import init_database
from supply_chain_agents.workflow_state import WorkflowState


def _make_requested_item(
    item_description: str = "A4 paper",
    requested_quantity: int = 1,
    **overrides,
) -> RequestedItem:
    item_fields = {
        "order_date": date(2026, 1, 1),
        "delivery_date": date(2026, 1, 15),
    }
    item_fields.update(overrides)
    return RequestedItem(
        item_description=item_description,
        requested_quantity=requested_quantity,
        **item_fields,
    )


@pytest.fixture(autouse=True)
def workflow_state():
    # Initializing the workflow state
    workflow_state = WorkflowState(
        db_engine=test_db_engine,
        catalog_items=catalog_items,
        product_prices=product_prices,
        catalog_embeddings=np.empty((0, 0)),
    )
    return workflow_state


def test_add_invalid_quantity_item(workflow_state):
    with pytest.raises(ValidationError):
        add_order_item(
            item_description="A4 paper",
            requested_quantity=-12,
            order_date="2026-01-01",
            delivery_date="2026-01-15",
            workflow_state=workflow_state,
        )

    assert workflow_state.requested_items == []


def test_add_zero_quantity_item(workflow_state):
    with pytest.raises(ValidationError):
        add_order_item(
            item_description="A4 paper",
            requested_quantity=0,
            order_date="2026-01-01",
            delivery_date="2026-01-15",
            workflow_state=workflow_state,
        )

    assert workflow_state.requested_items == []


def test_add_invalid_order_date_item(workflow_state):
    with pytest.raises(ValueError):
        add_order_item(
            item_description="A4 paper",
            requested_quantity=12,
            order_date="01-01-2026",
            delivery_date="2026-01-15",
            workflow_state=workflow_state,
        )

    assert workflow_state.requested_items == []


def test_add_invalid_delivery_date_item(workflow_state):
    with pytest.raises(ValueError):
        add_order_item(
            item_description="A4 paper",
            requested_quantity=12,
            order_date="2026-01-01",
            delivery_date="15-01-2026",
            workflow_state=workflow_state,
        )

    assert workflow_state.requested_items == []


def test_add_valid_item(workflow_state):
    result = add_order_item(
        item_description="A4 paper",
        requested_quantity=33,
        order_date="2026-01-01",
        delivery_date="2026-01-15",
        workflow_state=workflow_state,
    )
    assert len(workflow_state.requested_items) == 1
    added_item = workflow_state.requested_items[0]
    assert isinstance(added_item.order_line_id, str)
    assert added_item.item_description == "A4 paper"
    assert added_item.requested_quantity == 33
    assert added_item.can_fulfill is False
    assert added_item.shortage_quantity == 0
    assert added_item.supplier_order_feasible is False
    assert added_item.supplier_delivery_date is None
    assert added_item.order_date == date(2026, 1, 1)
    assert added_item.delivery_date == date(2026, 1, 15)
    assert added_item.discount_rate == 0
    assert added_item.sale_completed is False
    assert json.loads(result) == added_item.model_dump(mode="json")


def test_assign_inventory_name(workflow_state):
    workflow_state.catalog_embeddings = embed(workflow_state.catalog_items)
    item_1 = _make_requested_item(
        item_description="A4 printer paper",
        requested_quantity=33,
        order_date=date(2026, 1, 1),
        delivery_date=date(2026, 1, 20),
    )

    item_2 = _make_requested_item(
        item_description="balloons",
        requested_quantity=33,
        order_date=date(2026, 1, 1),
        delivery_date=date(2026, 1, 20),
    )

    item_3 = _make_requested_item(
        item_description="decorative washi tape",
        requested_quantity=33,
        order_date=date(2026, 1, 1),
        delivery_date=date(2026, 1, 20),
    )

    workflow_state.requested_items = [item_1, item_2, item_3]
    assign_inventory_name(workflow_state=workflow_state)
    assert item_1.matched_catalog_item == "A4 paper"
    assert item_2.matched_catalog_item is None
    assert item_3.matched_catalog_item == "Decorative adhesive tape (washi tape)"


def test_check_stock_inventory(workflow_state):
    init_database(workflow_state.db_engine)
    item_1 = _make_requested_item(
        item_description="Construction paper",
        matched_catalog_item="Construction paper",
        requested_quantity=200,
        order_date=date(2026, 1, 1),
        delivery_date=date(2026, 1, 20),
    )

    item_2 = _make_requested_item(
        item_description="Streamers",
        matched_catalog_item="Party streamers",
        requested_quantity=500,
        order_date=date(2026, 1, 1),
        delivery_date=date(2026, 1, 30),
    )
    workflow_state.requested_items = [item_1, item_2]
    check_stock_inventory(workflow_state=workflow_state)

    assert item_1.can_fulfill is True
    assert item_1.shortage_quantity == 0

    assert item_2.can_fulfill is False
    assert item_2.shortage_quantity == 283


def test_check_delivery_timeline(workflow_state):
    init_database(workflow_state.db_engine)
    item = _make_requested_item(
        item_description="Streamers",
        matched_catalog_item="Party streamers",
        requested_quantity=500,
        order_date=date(2026, 1, 1),
        delivery_date=date(2026, 1, 5),
        shortage_quantity=283,
    )
    workflow_state.requested_items.append(item)

    check_delivery_timeline(workflow_state=workflow_state)
    assert item.supplier_order_feasible is True
    assert item.supplier_delivery_date == date(2026, 1, 5)
    item.delivery_date = date(2026, 1, 4)
    check_delivery_timeline(workflow_state=workflow_state)
    assert item.supplier_order_feasible is False
    assert item.supplier_delivery_date is None


def test_execute_sale_completes_sale_and_applies_discount(workflow_state, monkeypatch):
    item = _make_requested_item(
        requested_quantity=3,
        matched_catalog_item="A4 paper",
        discount_rate=0.1,
    )
    workflow_state.requested_items = [item]
    transaction_calls = []

    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pd.DataFrame({"current_stock": [10]}),
    )

    def create_transaction(db_engine, inventory_name, **kwargs):
        transaction_calls.append((db_engine, inventory_name, kwargs))
        return "transaction-id"

    monkeypatch.setattr(order_services, "create_transaction", create_transaction)

    execute_sale(workflow_state)

    assert item.sale_completed is True
    assert item.can_fulfill is True
    assert item.shortage_quantity == 0
    assert item.order_line_id in workflow_state.sold_order_line_ids
    assert transaction_calls == [
        (
            workflow_state.db_engine,
            "A4 paper",
            {
                "transaction_type": "sales",
                "quantity": 3,
                "price": round(workflow_state.product_prices["A4 paper"] * 3 * 0.9, 2),
                "date": item.order_date,
            },
        )
    ]


def test_execute_sale_reorders_shortage_before_selling(workflow_state, monkeypatch):
    item = _make_requested_item(
        requested_quantity=75,
        matched_catalog_item="Party streamers",
        can_fulfill=False,
        shortage_quantity=25,
        supplier_order_feasible=True,
    )
    workflow_state.requested_items = [item]
    stock_levels = iter([50, 75])
    reorder_calls = []
    transaction_calls = []

    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pd.DataFrame({"current_stock": [next(stock_levels)]}),
    )

    def reorder_supply(**kwargs):
        reorder_calls.append(kwargs)
        return True

    def create_transaction(db_engine, inventory_name, **kwargs):
        transaction_calls.append((inventory_name, kwargs))
        return "transaction-id"

    monkeypatch.setattr(order_services, "reorder_supply", reorder_supply)
    monkeypatch.setattr(order_services, "create_transaction", create_transaction)

    execute_sale(workflow_state)

    assert reorder_calls == [
        {
            "inventory_name": "Party streamers",
            "quantity": 25,
            "order_date": item.order_date,
            "workflow_state": workflow_state,
        }
    ]
    assert transaction_calls[0][1]["transaction_type"] == "sales"
    assert item.sale_completed is True
    assert item.can_fulfill is True


def test_execute_sale_skips_unmatched_invalid_and_already_sold_items(
    workflow_state, monkeypatch
):
    unmatched_item = _make_requested_item(matched_catalog_item=None)
    invalid_item = _make_requested_item(
        item_description="unknown", matched_catalog_item="not in catalog"
    )
    sold_item = _make_requested_item()
    workflow_state.requested_items = [unmatched_item, invalid_item, sold_item]
    workflow_state.sold_order_line_ids.add(sold_item.order_line_id)

    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pytest.fail("stock should not be checked for skipped items"),
    )
    monkeypatch.setattr(
        order_services,
        "create_transaction",
        lambda *args, **kwargs: pytest.fail("skipped items must not be sold"),
    )

    execute_sale(workflow_state)

    assert all(item.sale_completed is False for item in workflow_state.requested_items)


def test_execute_sale_aborts_when_shortage_does_not_match_stock(
    workflow_state, monkeypatch
):
    item = _make_requested_item(
        requested_quantity=10,
        matched_catalog_item="A4 paper",
        shortage_quantity=4,
        supplier_order_feasible=True,
    )
    workflow_state.requested_items = [item]

    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pd.DataFrame({"current_stock": [5]}),
    )
    monkeypatch.setattr(
        order_services,
        "reorder_supply",
        lambda **kwargs: pytest.fail("mismatched shortage must abort before reorder"),
    )

    execute_sale(workflow_state)

    assert item.sale_completed is False
    assert item.order_line_id not in workflow_state.sold_order_line_ids


def test_execute_sale_stops_when_supplier_reorder_fails(workflow_state, monkeypatch):
    item = _make_requested_item(
        requested_quantity=10,
        matched_catalog_item="A4 paper",
        shortage_quantity=5,
        supplier_order_feasible=True,
    )
    workflow_state.requested_items = [item]
    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pd.DataFrame({"current_stock": [5]}),
    )
    monkeypatch.setattr(order_services, "reorder_supply", lambda **kwargs: False)
    monkeypatch.setattr(
        order_services,
        "create_transaction",
        lambda *args, **kwargs: pytest.fail("sale must not occur after reorder failure"),
    )

    execute_sale(workflow_state)

    assert item.sale_completed is False
    assert item.can_fulfill is False
    assert item.order_line_id not in workflow_state.sold_order_line_ids


def test_execute_sale_stops_when_stock_is_insufficient(workflow_state, monkeypatch):
    item = _make_requested_item(
        requested_quantity=10, matched_catalog_item="A4 paper"
    )
    workflow_state.requested_items = [item]
    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pd.DataFrame({"current_stock": [5]}),
    )
    monkeypatch.setattr(
        order_services,
        "create_transaction",
        lambda *args, **kwargs: pytest.fail("sale must not occur without enough stock"),
    )

    execute_sale(workflow_state)

    assert item.sale_completed is False
    assert item.can_fulfill is False
    assert item.order_line_id not in workflow_state.sold_order_line_ids


def test_execute_sale_resets_state_and_reraises_transaction_error(
    workflow_state, monkeypatch
):
    item = _make_requested_item(requested_quantity=2, matched_catalog_item="A4 paper")
    item.can_fulfill = True
    workflow_state.requested_items = [item]
    monkeypatch.setattr(
        order_services,
        "get_stock_level",
        lambda *args: pd.DataFrame({"current_stock": [10]}),
    )

    def fail_transaction(*args, **kwargs):
        raise RuntimeError("transaction failed")

    monkeypatch.setattr(order_services, "create_transaction", fail_transaction)

    with pytest.raises(RuntimeError, match="transaction failed"):
        execute_sale(workflow_state)

    assert item.sale_completed is False
    assert item.can_fulfill is False
    assert item.order_line_id not in workflow_state.sold_order_line_ids


def test_execute_restock_reorders_only_items_below_minimum(
    workflow_state, monkeypatch
):
    workflow_state.min_stock_levels = {
        "A4 paper": 5,
        "Construction paper": 5,
        "Party streamers": 7,
    }
    monkeypatch.setattr(
        order_services,
        "get_all_inventory",
        lambda **kwargs: {
            "A4 paper": 3,
            "Construction paper": 1,
            "Party streamers": 8,
            "unconfigured item": 0,
        },
    )
    reorder_calls = []

    def reorder_supply(**kwargs):
        reorder_calls.append(kwargs)
        return kwargs["inventory_name"] == "A4 paper"

    monkeypatch.setattr(order_services, "reorder_supply", reorder_supply)

    result = execute_restock("2026-01-01", workflow_state)

    assert result == "Following items got restocked: A4 paper"
    assert reorder_calls == [
        {
            "inventory_name": "A4 paper",
            "quantity": 2,
            "order_date": date(2026, 1, 1),
            "workflow_state": workflow_state,
        },
        {
            "inventory_name": "Construction paper",
            "quantity": 4,
            "order_date": date(2026, 1, 1),
            "workflow_state": workflow_state,
        },
    ]


def test_execute_restock_reports_when_no_restocking_is_needed(
    workflow_state, monkeypatch
):
    workflow_state.min_stock_levels = {"A4 paper": 5}
    monkeypatch.setattr(
        order_services,
        "get_all_inventory",
        lambda **kwargs: {"A4 paper": 5},
    )
    monkeypatch.setattr(
        order_services,
        "reorder_supply",
        lambda **kwargs: pytest.fail("inventory at minimum should not be reordered"),
    )

    result = execute_restock("2026-01-01", workflow_state)

    assert result == "All items are available - no restocking performed"


def test_get_financial_status_updates_report(workflow_state, monkeypatch):
    monkeypatch.setattr(
        order_services,
        "generate_financial_report",
        lambda db_engine, as_of_date: {
            "as_of_date": as_of_date,
            "cash_balance": 1250.5,
            "inventory_value": 300.25,
            "total_assets": 1550.75,
        },
    )

    result = get_financial_status("2026-01-01", workflow_state)

    assert json.loads(result) == {
        "as_of_date": "2026-01-01",
        "cash_balance": 1250.5,
        "inventory_value": 300.25,
    }
    assert workflow_state.financial_report.model_dump(mode="json") == json.loads(result)


def test_get_financial_status_returns_validation_error(workflow_state, monkeypatch):
    monkeypatch.setattr(
        order_services,
        "generate_financial_report",
        lambda db_engine, as_of_date: {
            "as_of_date": as_of_date,
            "cash_balance": "not a number",
            "inventory_value": 300.25,
        },
    )

    result = get_financial_status("2026-01-01", workflow_state)

    assert result.startswith("Validation error:")
    assert result.endswith(" - Financial report generation failed")
    assert workflow_state.financial_report is None
