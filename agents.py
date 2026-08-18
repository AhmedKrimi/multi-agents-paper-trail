from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from items_model import RequestedItem
from pydantic import ValidationError
from config import item_schema
import uuid
import json
from context import Context
from utils import (
    validate_item,
    create_transaction,
    get_stock_level,
    get_supplier_delivery_date,
    _parse_date,
    search_quote_history,
    reorder_supply,
)


# Set up the different agents
class OrderProcessorAgent(ToolCallingAgent):
    "Agent responsible for extracting information from customers requests"

    def __init__(self, model: OpenAIServerModel):
        # Tools for order processor
        @tool
        def assign_item(
            item_info: str, requested_quantity: int, order_date: str, delivery_date: str
        ) -> str:
            """Assign an unique id to each item requested
            Args:
                item_info (str): all info of the item in the customer request
                requested_quantity (int): requested_quantity of the item in the customer request
                order_date (str): date of the order by the customer in <YYYY-MM-DD> format
                delivery_date (str): delivery date expected of the item in the customer request in <YYYY-MM-DD>
            Returns:
                Item (str) extracted from the customer request
            """
            item = RequestedItem(
                suborder_id=str(uuid.uuid4()),
                item_info=item_info,
                inventory_name="N/A",
                requested_quantity=requested_quantity,
                can_fulfil=False,
                short_by=0,
                order_supplier=False,
                estimated_date="N/A",
                order_date=order_date,
                delivery_date=delivery_date,
                discount=0.0,
                suborder_executed=False,
            )
            return item.model_dump_json()

        super().__init__(
            tools=[assign_item],
            model=model,
            name="order_processor",
            description="""
                Extract order details from a request
             """,
        )


class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing the inventory"""

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="inventory_manager",
            description="Responsible for checking the inventory if all items are available in stock",
        )

    def _build_tools(self):
        ctx = self.ctx

        # Tools for inventory agent
        @tool
        def assign_inventory_name(item_dict: dict, valid_name: str) -> str:
            """Assign the requested item a valid inventory name
            Args:
                item_dict (dict): dict containing all the information of the item requested
                valid_name (str): Closest name found in available items to the requested item
            Returns:
                item (str): Updated requested item with valid name from the inventory
            """
            try:
                item = validate_item(item_dict)
            except ValidationError as e:
                return f"ERROR: Validation error: {e}"
            if valid_name not in ctx.available_items:
                item.inventory_name = "N/A"
            else:
                item.inventory_name = valid_name

            return item.model_dump_json()

        @tool
        def check_inventory(item_dict: dict) -> str:
            """Check whether an item is in stock in sufficient quantity
            Args:
                item_dict (dict): dict containing all the information of the item requested
            Returns:
                item (str): Updated requested item with item availability in the inventory
            """

            try:
                item = validate_item(item_dict)
            except ValidationError as e:
                return f"ERROR: Validation error: {e}"
            inventory_name = item.inventory_name
            order_date = item.order_date
            quantity = item.requested_quantity
            if inventory_name not in ctx.available_items:
                return (
                    f"'{inventory_name}' is not a valid catalog name.- CANNOT FULFILL"
                )
            current_stock = int(
                get_stock_level(ctx.db_engine, inventory_name, order_date)[
                    "current_stock"
                ].iloc[0]
            )
            if current_stock >= quantity:
                item.can_fulfil = True
            else:
                item.can_fulfil = False
                item.short_by = quantity - current_stock
            return item.model_dump_json()

        @tool
        def check_delivery_timeline(item_dict: dict) -> str:
            """Check if the quantity of an item can be delivered by a supplier before the delivery_date
            Args:
                item_dict (dict): dict containing all the information of the item requested
            Returns:
                item (str): Updated requested item with information about estimated date of supplier delivery
            """
            try:
                item = validate_item(item_dict)
            except ValidationError as e:
                return f"ERROR: Validation error: {e}"
            order_date = item.order_date
            delivery_date = item.delivery_date
            quantity = item.requested_quantity

            estimated_date = get_supplier_delivery_date(
                input_date_str=order_date, quantity=quantity
            )

            if _parse_date(estimated_date) > _parse_date(delivery_date):
                item.order_supplier = False
            else:
                item.order_supplier = True
                item.estimated_date = estimated_date
            return item.model_dump_json()

        return [assign_inventory_name, check_inventory, check_delivery_timeline]


class QuoteManagerAgent(ToolCallingAgent):
    "Agent responsible for pricing of the goods"

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="quote_manager",
            description="responsible for pricing of the goods",
        )

    def _build_tools(self):
        ctx = self.ctx

        @tool
        def calculate_discount(item_dict: dict) -> str:
            """Apply a discount to the price of the item
            Args:
                item_dict (dict): Dict containing all the information of the item requested
            Returns:
                item (str): Updated requested item with the discount to be applied at checkout
            """
            try:
                item = validate_item(item_dict)
            except ValidationError as e:
                return f"ERROR: Validation error: {e}"
            discount = None
            res = search_quote_history(
                ctx.db_engine, [item.item_info], limit=1)

            if len(res) > 0:
                order_size = res[0]["order_size"]
                if order_size == "large":
                    discount = 0.2
                elif order_size == "medium":
                    discount = 0.1
                else:
                    discount = 0.0
            else:
                print(
                    "INFO: No history found for this request, calculating the order_size...."
                )
                quantity = item.requested_quantity
                if quantity >= 1000:  # Large
                    discount = 0.2
                elif quantity >= 100:  # Medium
                    discount = 0.1
                else:
                    discount = 0.0  # Small

            # update item discount
            item.discount = discount
            return item.model_dump_json()

        return [calculate_discount]


class SalesManagerAgent(ToolCallingAgent):
    "Agent responsible for executing orders"

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.ctx = ctx
        super().__init__(
            tools=self._build_tools(),
            model=model,
            name="sales_manager",
            description="""
                Agent responsible for executing sales after it was processed by inventory manager
            """,
        )

    def _build_tools(self):
        ctx = self.ctx

        # Tools for the sales agent
        @tool
        def execute_sale(item_dict: dict) -> str:
            """Execute transaction for each item request by the customer
            Args:
                item_dict (dict): Dict containing all the information of the item requested
            Returns:
                item (str): Updated requested item with information about the sale execution
            """
            try:
                item = validate_item(item_dict)
            except ValidationError as e:
                return f"ERROR: Validation error: {e}"
            item_id = item.suborder_id
            inventory_name = item.inventory_name
            quantity = item.requested_quantity
            order_date = item.order_date
            order_supplier = item.order_supplier
            short_by = item.short_by
            discount = item.discount

            if item_id in ctx.items_sold:
                return f"{item_id} selling transaction has already executed, skip to the next item"

            if inventory_name not in ctx.available_items:
                return f"transaction for {inventory_name} has failed"

            transaction_date = order_date

            # Order the missing quantity from the supplier first
            if order_supplier:
                actual_short_by = quantity - int(
                    get_stock_level(ctx.db_engine, inventory_name, order_date)[
                        "current_stock"
                    ].iloc[0]
                )
                if actual_short_by != short_by:
                    print(
                        f"ERROR: there is a mismatch between actual short by {actual_short_by} and the short by calculated by the inventory manager: {short_by}, sell aborted"
                    )
                    item.suborder_executed = False
                    return item.model_dump_json()
                res = reorder_supply(
                    inventory_name=inventory_name,
                    quantity=short_by,
                    order_date=transaction_date,
                    ctx=ctx,
                )
                if not res:
                    print(
                        f"ERROR: Ordering {inventory_name} with {short_by} has failed!"
                    )
                    item.suborder_executed = False
                    return item.model_dump_json()

            # Check first if the quantity is available before the delivery date
            current_stock = int(
                get_stock_level(ctx.db_engine, inventory_name, transaction_date)[
                    "current_stock"
                ].iloc[0]
            )
            if current_stock < quantity:
                print(
                    f"ERROR: Selling {inventory_name} failed! quantity in stock: {current_stock}"
                    f" is not enough to cover requested quantity: {quantity}"
                )
                item.suborder_executed = False
            else:
                item.can_fulfil = True

            # Execute the sale
            if item.can_fulfil:
                unit_price = ctx.products_price[inventory_name]
                price = round((1 - discount) * unit_price * quantity, 2)

                try:
                    res = create_transaction(
                        ctx.db_engine,
                        inventory_name,
                        transaction_type="sales",
                        quantity=quantity,
                        price=price,
                        date=transaction_date,
                    )
                    ctx.items_sold.add(item_id)
                    item.suborder_executed = True
                    print(
                        f"INFO: Selling {inventory_name} transaction was successful - price: {price} - transaction ID: {res}"
                    )
                except Exception as e:
                    item.suborder_executed = False
                    print(
                        f"Selling {inventory_name} transaction has failed : {e}")

            return item.model_dump_json()

        return [execute_sale]


class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates the activities of all agents"""

    def __init__(self, model: OpenAIServerModel, ctx: Context):
        self.model = model
        self.ctx = ctx
        # Initialize specialized agents
        self.inventory_manager = InventoryManagerAgent(model, ctx)
        self.order_processor = OrderProcessorAgent(model)
        self.quote_manager = QuoteManagerAgent(model, ctx)
        self.sales_manager = SalesManagerAgent(model, ctx)
        self.order_processor_resp = None
        self.inventory_manager_resp = None
        self.quote_manager_resp = None

        @tool
        def get_order_details(request_w_date: str) -> str:
            """Extract relevant information from customer message
            Args:
                request_w_date: request made by the customer with the date

            Return:
                All relevant information about the request: inventory_name, quantity, request date, requested delivery date
            """
            self.order_processor_resp = self.order_processor.run(f"""
            Customer request: {request_w_date}\n"
            Extract every item requested and call assign_item to assign an item object to it. the result of the extraction must be a list of this schema {json.dumps(item_schema)}\n"
            Call final_answer: List of the items in this schema: {json.dumps(item_schema)}
            """
            )
            return self.order_processor_resp

        @tool
        def manage_inventory() -> str:
            """check if the items are in the inventory or they need to be
            reordered from the supplier
            Returns
                List of available items that can be delivered
            """
            if self.order_processor_resp is None:
                return "Order was not processed yet, call get_order_details first"

            self.inventory_manager_resp = self.inventory_manager.run(f"""
            Valid catalog item names are exactly: {", ".join(ctx.available_items)}
            For each item in {self.order_processor_resp}
             1 - Map the customer's wording to the closest valid catalog name above, call assign_inventory_name with the result
             2 - Call check_inventory
             3-  Call check_delivery_timeline
            When every item has been checked, and all tool calls finish, then call final_answer tool:
             final_answer: List of the items in this schema: {json.dumps(item_schema)}
            """
            )
            return self.inventory_manager_resp

        @tool
        def prepare_quote() -> str:
            """Prepare quote for the order requested by the customer
            Returns:
                Determines whether a discount applies to the items to increase client satisfaction.
            """
            if self.inventory_manager_resp is None:
                return "Items are not checked yet by the inventory! Call manage_inventory first"

            self.quote_manager_resp = self.quote_manager.run(f"""
            1 - Call calculate_discount for each item in the List:{self.inventory_manager_resp}
            2 - Call final_answer: List of the items with this schema: {json.dumps(item_schema)}
            """)
            return self.quote_manager_resp

        @tool
        def prepare_sale() -> str:
            """Execute the orders from sale_details checked by the inventory manager
            Returns:
                Final message whether the sale is executed or not
            """
            if self.quote_manager_resp is None:
                return "No Quote has been prepared yet! Call prepare_quote first"

            # keep track of items sold:
            self.ctx.items_sold.clear()

            return self.sales_manager.run(f"""
            1 - Call execute_sale for each item in the list: {self.quote_manager_resp}:
            2 - When every item has been checked and all tool calls finish, then call final_answer tool alone with:
             List of the items in this schema: {json.dumps(item_schema)}
             Then a final line:
              VERDICT: ORDER FULFILLED
              OR
              VERDICT: ORDER is fulfilled partially, <list the items that were sold successfully>
              OR
              VERDICT: ORDER cannot be fulfilled
            Do not add any other text.
            """
                                          )

        # 2 - Update the inventory if the sale transaction is a success

        super().__init__(
            model=model,
            tools=[get_order_details, manage_inventory,
                   prepare_quote, prepare_sale],
            name="orchestrator",
            description="""
                        You are the orchestrator for a paper company
                        You coordinate between the order processor, inventory manager, quote manager and sales manager agents.
                        """,
        )

    def process_customer_order(self, customer_message: str) -> str:
        """Process a customer order through the coordinated agent workflow.

        Args:
            customer_message: The customer's order request

        Returns:
            provide the answer as last answer
        """

        print("\n--- Processing New Order ---")

        # Use the orchestrator's own coordination workflow
        context = f"""
        Customer request: "{customer_message}"
        IMPORTANT: DO NOT CREATE FAKE TRANSACTIONS/DATES/ORDERS, IF SOMETHING WENT WRONG OR MISSING, HIGHLIGHT IT
        Process this order by coordinating with our specialized agents:
        For customer orders, follow this workflow:
         1. Call get_order_details with the customer request to extract the items.
         2. Call manage_inventory with the extracted item list
         3. call prepare_quote with the result of manage_inventory
         4. Call prepare_sale with the result of prepare_quote
         5. Call final_answer based on the verdict of sale managers agent:
          - if all items have suborder_executed : True, then inform the customer that the request will be fulfilled and give information about the delivery date of each item from prepare_sale
          - if some items have suborder_executed: False, then inform the customer that the request will be partially fulfilled and provide them information about the items that cannot be shipped and the ones that will be with the delivery date of each item from prepare_sale.
          - if all items have suborder_executed: False, then inform the customer that none of the items are available and the request cannot be fulfilled
        """
        return self.run(task=context)
