import numpy as np
from pydantic import ValidationError
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

from config import MATCHING_THRESHOLD
from config_logging import get_logger
from context import Context
from models import FinancialReport, RequestedItem
from utils import (
    _parse_date,
    apply_bulk_discount,
    create_transaction,
    embed,
    generate_financial_report,
    get_all_inventory,
    get_stock_level,
    get_supplier_delivery_date,
    reorder_supply,
    return_full_state,
    search_quote_history,
    validate_model,
)


# Set up the different agents
class OrderProcessorAgent(ToolCallingAgent):
    """Agent responsible for extracting information from customer requests"""

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        self.logger = get_logger(self)
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="order_processor",
            description="Extract order details from a request",
        )

    def _build_tools(self):
        ctx = self.ctx
        logger = self.logger

        @tool
        def extract_item_description(
            item_description: str,
            requested_quantity: int,
            order_date: str,
            delivery_date: str,
        ) -> str:
            """Extract item info from the customer request
            Args:
                item_description (str):  original item description extracted from the customer request
                requested_quantity (int): requested_quantity of the item in the customer request
                order_date (str): date of the order by the customer in <YYYY-MM-DD> format
                delivery_date (str): delivery date expected of the item in the customer request in <YYYY-MM-DD>
            Returns:
                str: JSON string representing the RequestedItem with their extracted item_description, requested_quantity, order_date and delivery_date
            """

            item = RequestedItem(
                item_description=item_description,
                requested_quantity=requested_quantity,
                order_date=_parse_date(order_date),
                delivery_date=_parse_date(delivery_date),
            )
            ctx.requested_items.append(item)

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

        return [extract_item_description]


class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing the inventory"""

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        self.logger = get_logger(self)
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="inventory_manager",
            description="Checks whether requested items are available in stock",
        )

    def _build_tools(self):
        ctx = self.ctx
        logger = self.logger

        # Tools for inventory agent
        @tool
        def assign_inventory_name() -> str:
            """Match each requested item to a valid catalog item.

            Returns:
                str: List of the RequestedItems with their updated matched_catalog_item closest to item_description
            """

            for item in ctx.requested_items:
                query_vec = embed(item.item_description)
                cos_sims = ctx.catalog_embeddings @ query_vec
                best_idx = int(np.argmax(cos_sims))
                matched_catalog_item = (
                    ctx.catalog_items[best_idx]
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

            return return_full_state(ctx.requested_items)

        @tool
        def check_stock_inventory() -> str:
            """Check whether the current requested items are in sufficient quantity in the inventory
            Returns:
                str: List of requested items with their status in the inventory (can_fulfill, shortage_quantity) updated
            """

            for item in ctx.requested_items:
                matched_catalog_item = item.matched_catalog_item
                order_date = item.order_date
                quantity = item.requested_quantity
                if matched_catalog_item not in ctx.catalog_items:
                    item.can_fulfill = False
                    continue
                current_stock = int(
                    get_stock_level(ctx.db_engine, matched_catalog_item, order_date)[
                        "current_stock"
                    ].iloc[0]
                )
                if current_stock >= quantity:
                    item.can_fulfill = True
                else:
                    item.can_fulfill = False
                    item.shortage_quantity = quantity - current_stock

            return return_full_state(items=ctx.requested_items)

        @tool
        def check_delivery_timeline() -> str:
            """Check if the quantity of an item can be delivered by a supplier before the delivery_date
            Returns:
                str: list of all requested items (with supplier_order_feasible/supplier_delivery_date updated)
            """
            for item in ctx.requested_items:
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
                    else:
                        item.supplier_order_feasible = True
                        item.supplier_delivery_date = _parse_date(
                            supplier_delivery_date
                        )
            return return_full_state(items=ctx.requested_items)

        return [assign_inventory_name, check_stock_inventory, check_delivery_timeline]


class QuoteManagerAgent(ToolCallingAgent):
    """Agent responsible for calculating item discounts and pricing"""

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        self.logger = get_logger(self)
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="quote_manager",
            description="Agent responsible for calculating item discounts and pricing",
        )

    def _build_tools(self):
        ctx = self.ctx
        logger = self.logger

        @tool
        def calculate_discount_rate() -> str:
            """Calculate and apply a bulk discount to the customer order

            Returns:
                str: List of requested items with the calculated discount_rate
            """
            discount_rate = 0.0
            quote_history = search_quote_history(
                ctx.db_engine,
                [item.item_description for item in ctx.requested_items],
                limit=1,
            )

            if quote_history:
                order_size = quote_history[0]["order_size"]
                if order_size == "large":
                    discount_rate = 0.2
                elif order_size == "medium":
                    discount_rate = 0.1
                else:
                    discount_rate = 0.0
            else:
                logger.debug(
                    "No history found for this request, calculating the order_size"
                )

                total_quantity = sum(
                    item.requested_quantity for item in ctx.requested_items
                )
                if total_quantity >= 1000:  # Large
                    discount_rate = 0.2
                elif total_quantity >= 100:  # Medium
                    discount_rate = 0.1
                else:
                    discount_rate = 0.0  # Small
            # Apply the calculated bulk discount to all the items
            apply_bulk_discount(ctx.requested_items, discount=discount_rate)

            return return_full_state(items=ctx.requested_items)

        return [calculate_discount_rate]


class SalesManagerAgent(ToolCallingAgent):
    "Agent responsible for executing orders, and performing after-sales operations, including restocking and financial reporting"

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        self.logger = get_logger(self)
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="sales_manager",
            description="""Agent responsible for executing sales after inventory and quote processing,
            and for post-sale operations including restocking and financial reporting
            """,
        )

    def _build_tools(self):
        ctx = self.ctx
        logger = self.logger

        # Tools for the sales agent
        @tool
        def execute_sale() -> str:
            """Execute a sale transaction for each requested item
            Returns:
                str: List of all items with their sale_completed attribute updated
            """
            for item in ctx.requested_items:
                if item.matched_catalog_item is None:
                    continue
                item_id = item.order_line_id
                matched_catalog_item = item.matched_catalog_item
                quantity = item.requested_quantity
                order_date = item.order_date
                supplier_order_feasible = item.supplier_order_feasible
                shortage_quantity = item.shortage_quantity
                discount_rate = item.discount_rate

                if item_id in ctx.sold_order_line_ids:
                    logger.warning(
                        "%s selling transaction has already executed, skip to the next item",
                        item_id,
                    )
                    continue

                if matched_catalog_item not in ctx.catalog_items:
                    logger.error("Transaction for %s has failed", matched_catalog_item)
                    continue

                transaction_date = order_date

                # Order the missing quantity from the supplier first
                if supplier_order_feasible:
                    actual_shortage_quantity = quantity - int(
                        get_stock_level(
                            ctx.db_engine, matched_catalog_item, order_date
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
                        order_date=transaction_date,
                        ctx=ctx,
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
                        ctx.db_engine, matched_catalog_item, transaction_date
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
                    unit_price = ctx.product_prices[matched_catalog_item]
                    price = round((1 - discount_rate) * unit_price * quantity, 2)

                    try:
                        transaction_id = create_transaction(
                            ctx.db_engine,
                            matched_catalog_item,
                            transaction_type="sales",
                            quantity=quantity,
                            price=price,
                            date=transaction_date,
                        )
                        ctx.sold_order_line_ids.add(item_id)
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

            return return_full_state(items=ctx.requested_items)

        @tool
        def execute_restock(order_date: str) -> str:
            """Restock inventory items below their minimum stock levels
            Args:
                order_date (str): order date of the request in <YYYY-MM-DD> format
            Returns:
                str: List of restocked items in the inventory (if any)
            """
            restocked_items = []
            inventory_snapshot = get_all_inventory(
                db_engine=ctx.db_engine, as_of_date=order_date
            )
            for item, stock in inventory_snapshot.items():
                if item not in ctx.min_stock_levels:
                    logger.warning(
                        "No minimum stock level configured for %s; skipping restock",
                        item,
                    )
                    continue
                min_stock_level = ctx.min_stock_levels[item]
                if stock < min_stock_level:
                    restock_quantity = min_stock_level - stock
                    logger.info(
                        "item: %s, is short in stock (only: %s left), restocking with quantity: %s",
                        item,
                        stock,
                        restock_quantity,
                    )
                    restock_succeeded = reorder_supply(
                        inventory_name=item,
                        quantity=restock_quantity,
                        order_date=_parse_date(order_date),
                        ctx=ctx,
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

        @tool
        def get_financial_status(order_date: str) -> str:
            """Get the cash balance and inventory value after sales and restocking
            Args:
                order_date (str): order date of the request in <YYYY-MM-DD> format
            Returns:
                str: Financial report containing the cash and the inventory status
            """
            financial_report_full = generate_financial_report(ctx.db_engine, order_date)
            keys = ["as_of_date", "cash_balance", "inventory_value"]
            financial_report_short = {k: financial_report_full[k] for k in keys}
            try:
                financial_report_model = validate_model(
                    FinancialReport, financial_report_short
                )
                ctx.financial_report = financial_report_model
                return ctx.financial_report.model_dump_json()
            except ValidationError as e:
                msg = f"Validation error: {e} - Financial report generation failed"
                logger.error(msg)
                return msg

        return [execute_sale, execute_restock, get_financial_status]


class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates the activities of all agents"""

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.model = model
        self.ctx = ctx
        self.orchestrator_logger = get_logger(self)
        # Initialize specialized agents
        self.inventory_manager = InventoryManagerAgent(model, ctx)
        self.order_processor = OrderProcessorAgent(model, ctx)
        self.quote_manager = QuoteManagerAgent(model, ctx)
        self.sales_manager = SalesManagerAgent(model, ctx)
        self.order_processor_resp = None
        self.inventory_manager_resp = None
        self.quote_manager_resp = None

        @tool
        def get_order_details(request_with_date: str) -> str:
            """Extract relevant information from customer message
            Args:
                request_with_date (str): request made by the customer with the date

            Returns:
                str: All relevant information about the request: item name (item_description), requested quantity, order date, delivery date
            """
            self.order_processor_resp = self.order_processor.run(f"""
            Customer request: {request_with_date}
            1. Extract every item requested and call extract_item_description to extract item_description, requested_quantity, order_date and delivery_date from the customer order
            2. call final_answer: list of extracted items from the order with all relevant information
            """)
            return self.order_processor_resp

        @tool
        def manage_inventory() -> str:
            """check if the items are in the inventory or they need to be
            reordered from the supplier
            Returns:
                str: Status of each requested item after the inventory check (availability and reorder needs)
            """
            if self.order_processor_resp is None:
                return "Order was not processed yet, call get_order_details first"

            self.inventory_manager_resp = self.inventory_manager.run("""
             1. call assign_inventory_name
             2. call check_stock_inventory
             3. call check_delivery_timeline
             4. call final_answer: the status of each requested item checked by the inventory manager
            """)
            return self.inventory_manager_resp

        @tool
        def prepare_quote() -> str:
            """Prepare quote for the order requested by the customer
            Returns:
                str: discount_rate calculated for each requested item
            """
            if self.inventory_manager_resp is None:
                return "Items are not checked yet by the inventory! call manage_inventory first"

            self.quote_manager_resp = self.quote_manager.run("""
            1. call calculate_discount_rate
            2. call final_answer: the discount_rate of each item requested by the customer
            """)
            return self.quote_manager_resp

        @tool
        def prepare_sale_restock_financial_report(request_date: str) -> str:
            """Execute the orders checked by the inventory manager, in addition to restocking and generating the financial report
            Args:
                request_date (str): Request date from the customer order in format YYYY-MM-DD
            Returns:
                str: Summary of which items were sold and which were rejected, plus the financial report containing the cash and the inventory status
            """
            if self.quote_manager_resp is None:
                return "No Quote has been prepared yet! call prepare_quote first"

            return self.sales_manager.run(f"""request_date: {request_date}
            1. call execute_sale
            2. call execute_restock with order_date from {request_date}.
            3. call get_financial_status with {request_date}.
            4. call final_answer:
                - status of the sale: which items have been sold and which ones got rejected
                - financial report (cash, and inventory status)
            """)

        @tool
        def get_order_state() -> str:
            """Return the current structured state of all requested order lines"""
            return return_full_state(self.ctx.requested_items)

        super().__init__(
            model=model,
            tools=[
                get_order_details,
                manage_inventory,
                prepare_quote,
                prepare_sale_restock_financial_report,
                get_order_state,
            ],
            name="orchestrator",
            description="""
            You are the orchestrator for Beaver's Choice Paper Company.

            Your role is to coordinate the order processor, inventory manager,
            quote manager, and sales manager to process a customer order from
            extraction through fulfillment and post-sale processing.
            """,
        )

    def process_customer_order(self, customer_message: str) -> str:
        """Process a customer order through the coordinated agent workflow.

        Args:
            customer_message (str): The customer's order request

        Returns:
            str: Message informing the customer about their order's execution and delivery timeline
        """

        self.orchestrator_logger.info("--- Processing New Order ---")

        self.reset_workflow()
        # Use the orchestrator's own coordination workflow
        coordination_workflow = f"""
        Customer request: "{customer_message}"
        As the orchestrator of the Beaver's Choice Paper Company, process this order by coordinating with the specialized agents:
        For customer orders, follow this workflow:
         1. call get_order_details with the customer request to extract the items.
         2. call manage_inventory
         3. call prepare_quote
         4. call prepare_sale_restock_financial_report
         5. call get_order_state
         6. call final_answer: review the list of RequestedItem state returned by the workflow.
            For each requested item:
            - If sale_completed=True, tell the customer the item was successfully processed and include its requested delivery_date.
            - If matched_catalog_item is None, explain that the requested item could not be matched to the catalog.
            - If sale_completed=False for another reason, give a short customer-facing explanation based on the workflow outcome without exposing internal system details.
            - If discount_rate > 0, mention the bulk discount once for the order as a percentage.
            - Use the customer's original item_description in the response.
            - Do not expose internal IDs, tool names, agent names, database errors, or implementation details.

            Then call final_answer with a concise, friendly customer-facing summary.
        """
        return self.run(task=coordination_workflow)

    def reset_workflow(self) -> None:
        """Reset workflow state before processing a new order"""
        self.order_processor_resp = None
        self.inventory_manager_resp = None
        self.quote_manager_resp = None
        self.ctx.financial_report = None
        self.ctx.sold_order_line_ids.clear()
        self.ctx.requested_items = []
