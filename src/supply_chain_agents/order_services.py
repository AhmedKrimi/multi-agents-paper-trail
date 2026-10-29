from logging import Logger

import numpy as np
from pydantic import ValidationError

from supply_chain_agents.config import MATCHING_THRESHOLD
from supply_chain_agents.config_logging import get_logger
from supply_chain_agents.embeddings import embed
from supply_chain_agents.models import FinancialReport, RequestedItem
from supply_chain_agents.utils import (
    _parse_date,
    apply_bulk_discount,
    create_transaction,
    generate_financial_report,
    get_all_inventory,
    get_stock_level,
    get_supplier_delivery_date,
    reorder_supply,
    return_full_state,
    validate_model,
)
from supply_chain_agents.workflow_state import WorkflowState


def add_order_item(
    item_description: str,
    requested_quantity: int,
    order_date: str,
    delivery_date: str,
    workflow_state: WorkflowState,
    logger: Logger | None = None,
) -> str:
    """Add an order item to the requested items in the shared workflow state

    Args:
        item_description (str):  original item description extracted from the customer request
        requested_quantity (int): requested_quantity of the item in the customer request
        order_date (str): date of the order by the customer in <YYYY-MM-DD> format
        delivery_date (str): delivery date expected of the item in the customer request in <YYYY-MM-DD>
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: JSON string representing the new RequestedItem with its extracted item_description, requested_quantity, order_date and delivery_date
    """
    logger = logger if logger is not None else get_logger(__name__)

    item = RequestedItem(
        item_description=item_description,
        requested_quantity=requested_quantity,
        order_date=_parse_date(order_date),
        delivery_date=_parse_date(delivery_date),
    )
    workflow_state.requested_items.append(item)

    logger.info(
        "item description: %s, requested quantity: %s"
        " order date: %s and delivery date: %s"
        " has been extracted from the customer order",
        item_description,
        requested_quantity,
        order_date,
        delivery_date,
    )

    return item.model_dump_json()


def assign_inventory_name(
    workflow_state: WorkflowState, logger: Logger | None = None
) -> str:
    """Match each requested item to a valid catalog item.
    Args:
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: List of the RequestedItems with their updated matched_catalog_item closest to item_description
    """
    logger = logger if logger is not None else get_logger(__name__)

    for item in workflow_state.requested_items:
        query_vec = embed(item.item_description)
        cos_sims = workflow_state.catalog_embeddings @ query_vec
        best_idx = int(np.argmax(cos_sims))
        matched_catalog_item = (
            workflow_state.catalog_items[best_idx]
            if cos_sims[best_idx] >= MATCHING_THRESHOLD
            else None
        )
        if matched_catalog_item:
            item.matched_catalog_item = matched_catalog_item
            logger.info(
                "Item: %s is assigned to the inventory name: %s, score: %s",
                item.item_description,
                matched_catalog_item,
                cos_sims[best_idx],
            )
        else:
            logger.warning(
                "Item %s could not be matched to the catalog",
                item.item_description,
            )

    return return_full_state(workflow_state.requested_items)


def check_stock_inventory(workflow_state: WorkflowState) -> str:
    """Check whether the current requested items are in sufficient quantity in the inventory
    Args:
        workflow_state (WorkflowState): Shared order state and database dependencies.
    Returns:
        str: List of requested items with their status in the inventory (can_fulfill, shortage_quantity) updated
    """

    for item in workflow_state.requested_items:
        matched_catalog_item = item.matched_catalog_item
        order_date = item.order_date
        quantity = item.requested_quantity
        if matched_catalog_item not in workflow_state.catalog_items:
            item.can_fulfill = False
            continue
        current_stock = int(
            get_stock_level(workflow_state.db_engine, matched_catalog_item, order_date)[
                "current_stock"
            ].iloc[0]
        )
        if current_stock >= quantity:
            item.can_fulfill = True
        else:
            item.can_fulfill = False
            item.shortage_quantity = quantity - current_stock

    return return_full_state(items=workflow_state.requested_items)


def check_delivery_timeline(
    workflow_state: WorkflowState, logger: Logger | None = None
) -> str:
    """Check if the quantity of an item can be delivered by a supplier before the delivery_date
    Args:
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: list of all requested items (with supplier_order_feasible/supplier_delivery_date updated)
    """
    logger = logger if logger is not None else get_logger(__name__)
    for item in workflow_state.requested_items:
        # Check if the item needs to be reordered from the supplier
        if item.matched_catalog_item is None:
            continue
        if not item.can_fulfill:
            order_date = item.order_date
            delivery_date = item.delivery_date
            shortage_quantity = item.shortage_quantity

            supplier_delivery_date = get_supplier_delivery_date(
                input_date_str=str(order_date), quantity=shortage_quantity
            )

            if _parse_date(supplier_delivery_date) > _parse_date(delivery_date):
                logger.warning(
                    "Item %s cannot be delivered on time from the supplier (supplier delivery date: %s)",
                    item.item_description,
                    str(supplier_delivery_date),
                )
                item.supplier_order_feasible = False
                item.supplier_delivery_date = None
            else:
                item.supplier_order_feasible = True
                item.supplier_delivery_date = _parse_date(supplier_delivery_date)
    return return_full_state(items=workflow_state.requested_items)


def calculate_discount_rate(
    workflow_state: WorkflowState, logger: Logger | None = None
) -> str:
    """Calculate and apply a bulk discount to the customer order
    Args:
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: List of requested items with the calculated discount_rate
    """
    logger = logger if logger is not None else get_logger(__name__)
    discount_rate = 0.0
    total_price = sum(
        item.requested_quantity
        * workflow_state.product_prices[item.matched_catalog_item]
        for item in workflow_state.requested_items
        if item.can_fulfill or item.supplier_order_feasible
    )
    if total_price >= 100:  # Large
        discount_rate = 0.2
    elif total_price >= 30:  # Medium
        discount_rate = 0.1
    else:
        discount_rate = 0.0  # Small
    # Apply the calculated bulk discount to all the items
    apply_bulk_discount(workflow_state.requested_items, discount=discount_rate)

    return return_full_state(items=workflow_state.requested_items)


def execute_sale(workflow_state: WorkflowState, logger: Logger | None = None) -> str:
    """Execute a sale transaction for each requested item
    Args:
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: List of all items with their sale_completed attribute updated
    """
    logger = logger if logger is not None else get_logger(__name__)
    for item in workflow_state.requested_items:
        if item.matched_catalog_item is None:
            continue
        item_id = item.order_line_id
        matched_catalog_item = item.matched_catalog_item
        quantity = item.requested_quantity
        order_date = item.order_date
        supplier_order_feasible = item.supplier_order_feasible
        shortage_quantity = item.shortage_quantity
        supplier_delivery_date = item.supplier_delivery_date
        discount_rate = item.discount_rate

        if item_id in workflow_state.sold_order_line_ids:
            logger.warning(
                "%s selling transaction has already executed, skip to the next item",
                item_id,
            )
            continue

        if matched_catalog_item not in workflow_state.catalog_items:
            logger.error("Transaction for %s has failed", matched_catalog_item)
            continue

        transaction_date = (
            supplier_delivery_date if supplier_order_feasible is True else order_date
        )

        # Order the missing quantity from the supplier first
        if supplier_order_feasible:
            actual_shortage_quantity = quantity - int(
                get_stock_level(
                    workflow_state.db_engine, matched_catalog_item, order_date
                )["current_stock"].iloc[0]
            )
            if actual_shortage_quantity != shortage_quantity:
                logger.error(
                    "Mismatch between actual short by %s and the short by calculated by the inventory manager: %s, sell aborted",
                    actual_shortage_quantity,
                    shortage_quantity,
                )
                item.sale_completed = False
                continue
            supplier_order_succeeded = reorder_supply(
                inventory_name=matched_catalog_item,
                quantity=shortage_quantity,
                arrival_date=supplier_delivery_date,
                workflow_state=workflow_state,
            )
            if not supplier_order_succeeded:
                logger.warning(
                    "Ordering %s with %s has failed!",
                    matched_catalog_item,
                    shortage_quantity,
                )
                item.sale_completed = False
                item.can_fulfill = False
                continue

        # Check first if the quantity is available before the delivery date
        current_stock = int(
            get_stock_level(
                workflow_state.db_engine, matched_catalog_item, transaction_date
            )["current_stock"].iloc[0]
        )
        if current_stock < quantity:
            logger.warning(
                "Selling %s failed! quantity in stock: %s"
                " is not enough to cover requested quantity: %s",
                matched_catalog_item,
                current_stock,
                quantity,
            )
            item.sale_completed = False
            item.can_fulfill = False
            continue
        else:
            item.shortage_quantity = 0
            item.can_fulfill = True

        # Execute the sale
        if item.can_fulfill:
            unit_price = workflow_state.product_prices[matched_catalog_item]
            price = round((1 - discount_rate) * unit_price * quantity, 2)

            try:
                transaction_id = create_transaction(
                    workflow_state.db_engine,
                    matched_catalog_item,
                    transaction_type="sales",
                    quantity=quantity,
                    price=price,
                    date=transaction_date,
                )
                workflow_state.sold_order_line_ids.add(item_id)
                item.sale_completed = True
                logger.info(
                    "Selling %s transaction was successful - price: %s - transaction ID: %s",
                    matched_catalog_item,
                    price,
                    transaction_id,
                )
            except Exception:
                item.sale_completed = False
                item.can_fulfill = False
                logger.exception(
                    "Selling %s transaction has failed", matched_catalog_item
                )
                raise

    return return_full_state(items=workflow_state.requested_items)


def execute_restock(
    order_date: str, workflow_state: WorkflowState, logger: Logger | None = None
) -> str:
    """Restock inventory items below their minimum stock levels
    Args:
        order_date (str): order date of the request in <YYYY-MM-DD> format
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: List of restocked items in the inventory (if any)
    """
    logger = logger if logger is not None else get_logger(__name__)
    restocked_items = []
    inventory_snapshot = get_all_inventory(
        db_engine=workflow_state.db_engine, as_of_date=order_date
    )
    for item, stock in inventory_snapshot.items():
        if item not in workflow_state.min_stock_levels:
            logger.warning(
                "No minimum stock level configured for %s; skipping restock",
                item,
            )
            continue
        min_stock_level = workflow_state.min_stock_levels[item]
        if stock < min_stock_level:
            restock_quantity = min_stock_level - stock
            logger.info(
                "item: %s, is short in stock (only: %s left), restocking with quantity: %s",
                item,
                stock,
                restock_quantity,
            )
            supplier_delivery_date = get_supplier_delivery_date(
                order_date, restock_quantity
            )
            restock_succeeded = reorder_supply(
                inventory_name=item,
                quantity=restock_quantity,
                arrival_date=_parse_date(supplier_delivery_date),
                workflow_state=workflow_state,
            )
            if restock_succeeded:
                restocked_items.append(item)
            else:
                logger.warning(
                    "Supplier reorder failed for %s, quantity: %s",
                    item,
                    restock_quantity,
                )
    if restocked_items:
        restocked_items_str = ", ".join(restocked_items)
        return f"Following items got restocked: {restocked_items_str}"
    return "All items are available - no restocking performed"


def get_financial_status(
    order_date: str, workflow_state: WorkflowState, logger: Logger | None = None
) -> str:
    """Get the cash balance and inventory value after sales and restocking
    Args:
        order_date (str): order date of the request in <YYYY-MM-DD> format
        workflow_state (WorkflowState): Shared order state and database dependencies.
        logger (Logger): Logger for operation outcomes and errors.
    Returns:
        str: Financial report containing the cash and the inventory status
    """
    logger = logger if logger is not None else get_logger(__name__)
    financial_report_full = generate_financial_report(
        workflow_state.db_engine, order_date
    )
    keys = ["as_of_date", "cash_balance", "inventory_value"]
    financial_report_short = {k: financial_report_full[k] for k in keys}
    try:
        financial_report_model = validate_model(FinancialReport, financial_report_short)
        workflow_state.financial_report = financial_report_model
        return workflow_state.financial_report.model_dump_json()
    except ValidationError as e:
        msg = f"Validation error: {e} - Financial report generation failed"
        logger.error(msg)
        return msg
