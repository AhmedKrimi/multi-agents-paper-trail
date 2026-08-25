from dataclasses import dataclass
from sqlalchemy import Engine


@dataclass
class Context:
    db_engine: Engine
    available_items: list[str]
    products_price: dict[str, float]
    min_stock_levels: dict[str, int]
    items_sold: set[str]
