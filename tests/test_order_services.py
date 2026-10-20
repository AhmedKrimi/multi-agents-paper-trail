import json
from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from supply_chain_agents.config import (
    catalog_items,
    product_prices,
    test_db_engine,
)
from supply_chain_agents.order_services import add_order_item
from supply_chain_agents.workflow_state import WorkflowState


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
            logger=None,
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
            logger=None,
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
            logger=None,
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
            logger=None,
        )

    assert workflow_state.requested_items == []


def test_add_valid_item(workflow_state):
    result = add_order_item(
        item_description="A4 paper",
        requested_quantity=33,
        order_date="2026-01-01",
        delivery_date="2026-01-15",
        workflow_state=workflow_state,
        logger=None,
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
