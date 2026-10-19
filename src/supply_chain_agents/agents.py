from smolagents import OpenAIServerModel, ToolCallingAgent, tool

from supply_chain_agents.config_logging import get_logger
from supply_chain_agents.context import Context
from supply_chain_agents.models import RequestedItem
from supply_chain_agents.order_services import (
    assign_inventory_name,
    calculate_discount_rate,
    check_delivery_timeline,
    check_stock_inventory,
    execute_restock,
    execute_sale,
    get_financial_status,
)
from supply_chain_agents.utils import (
    _parse_date,
    return_full_state,
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
        def assign_inventory_name_tool() -> str:
            """Match each requested item to a valid catalog item.

            Returns:
                str: List of the RequestedItems with their updated matched_catalog_item closest to item_description
            """
            return assign_inventory_name(ctx, logger)

        @tool
        def check_stock_inventory_tool() -> str:
            """Check whether the current requested items are in sufficient quantity in the inventory
            Returns:
                str: List of requested items with their status in the inventory (can_fulfill, shortage_quantity) updated
            """

            return check_stock_inventory(ctx)

        @tool
        def check_delivery_timeline_tool() -> str:
            """Check if the quantity of an item can be delivered by a supplier before the delivery_date
            Returns:
                str: list of all requested items (with supplier_order_feasible/supplier_delivery_date updated)
            """
            return check_delivery_timeline(ctx, logger)

        return [
            assign_inventory_name_tool,
            check_stock_inventory_tool,
            check_delivery_timeline_tool,
        ]


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
        def calculate_discount_rate_tool() -> str:
            """Calculate and apply a bulk discount to the customer order

            Returns:
                str: List of requested items with the calculated discount_rate
            """
            return calculate_discount_rate(ctx, logger)

        return [calculate_discount_rate_tool]


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
        def execute_sale_tool() -> str:
            """Execute a sale transaction for each requested item
            Returns:
                str: List of all items with their sale_completed attribute updated
            """
            return execute_sale(ctx, logger)

        @tool
        def execute_restock_tool(order_date: str) -> str:
            """Restock inventory items below their minimum stock levels
            Args:
                order_date (str): order date of the request in <YYYY-MM-DD> format
            Returns:
                str: List of restocked items in the inventory (if any)
            """
            return execute_restock(order_date, ctx, logger)

        @tool
        def get_financial_status_tool(order_date: str) -> str:
            """Get the cash balance and inventory value after sales and restocking
            Args:
                order_date (str): order date of the request in <YYYY-MM-DD> format
            Returns:
                str: Financial report containing the cash and the inventory status
            """
            return get_financial_status(order_date, ctx, logger)

        return [execute_sale_tool, execute_restock_tool, get_financial_status_tool]


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
