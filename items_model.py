from typing import Annotated

from pydantic import BaseModel, Field


class RequestedItem(BaseModel):
    suborder_id: Annotated[
        str, Field(description="ID of the suborder related to the requested item")
    ]
    item_info: Annotated[
        str, Field(description="All infos of the item from original order")
    ]
    inventory_name: Annotated[str, Field(description="Valid catalog name")]
    requested_quantity: Annotated[
        int, Field(ge=0, description="Quantity requested by the customer")
    ]
    can_fulfil: Annotated[
        bool,
        Field(
            description="Flag to indicate the quantity of the item can be sold from the inventory"
        ),
    ]
    short_by: Annotated[int, Field(description="Short by quantity from the inventory")]
    order_supplier: Annotated[
        bool,
        Field(
            description="Indicates whether the quantity unavailable in the stock can arrive before the delivery date"
        ),
    ]
    estimated_date: Annotated[
        str,
        Field(
            description="Estimated date to get the item from the supplier in format: <YYYY-MM-DD>"
        ),
    ]
    order_date: Annotated[str, Field(description="Order date in format: <YYYY-MM-DD>")]
    delivery_date: Annotated[
        str, Field(description="Delivery date in format: <YYYY-MM-DD>")
    ]
    discount: Annotated[float, Field(ge=0, le=1, description="Discount on the item")]
    suborder_executed: Annotated[
        bool,
        Field(
            description="Flag whether selling the item with the requested item was success or not"
        ),
    ]
