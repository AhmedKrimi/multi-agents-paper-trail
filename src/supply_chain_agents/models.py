import uuid
from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field


class RequestedItem(BaseModel):
    """Represents one customer order line throughout the processing workflow"""

    order_line_id: str = Field(
        description="Unique identifier for the customer order line",
        default_factory=lambda: uuid.uuid4().hex,
    )

    item_description: Annotated[
        str,
        Field(
            description="Original item description extracted from the customer order"
        ),
    ]
    matched_catalog_item: str | None = Field(
        description="Matched catalog item, if a valid match was found", default=None
    )
    requested_quantity: int = Field(
        ge=1, description="Quantity requested by the customer"
    )

    can_fulfill: bool = Field(
        description="Indicates whether the requested quantity can be delivered to the customer",
        default=False,
    )
    shortage_quantity: int = Field(
        description="Number of units needed to fulfill the requested quantity from inventory.",
        ge=0,
        default=0,
    )
    supplier_order_feasible: bool = Field(
        description="Indicates whether the quantity unavailable in the stock can arrive before the delivery date",
        default=False,
    )
    supplier_delivery_date: date | None = Field(
        description="Estimated date to get the item from the supplier in format: <YYYY-MM-DD>, if applicable",
        default=None,
    )
    order_date: Annotated[date, Field(description="Order date in format: <YYYY-MM-DD>")]
    delivery_date: Annotated[
        date, Field(description="Delivery date in format: <YYYY-MM-DD>")
    ]
    discount_rate: float = Field(
        ge=0, le=1, description="Discount rate applied to the requested item", default=0
    )
    sale_completed: bool = Field(
        description="Indicates whether the sale for the requested item was completed successfully.",
        default=False,
    )


class FinancialReport(BaseModel):
    """Summarizes cash and inventory value at a specific date"""

    as_of_date: Annotated[date, Field(description="The date of the report")]
    cash_balance: Annotated[float, Field(description="Total cash available")]
    inventory_value: Annotated[float, Field(description="Total value of inventory")]
