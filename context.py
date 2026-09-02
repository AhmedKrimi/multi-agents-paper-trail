from dataclasses import dataclass, field

from numpy import ndarray
from sqlalchemy import Engine

from models import FinancialReport, RequestedItem


@dataclass
class Context:
    """Shared state used by the agent workflow"""

    db_engine: Engine
    catalog_items: list[str]
    product_prices: dict[str, float]
    catalog_embeddings: ndarray
    min_stock_levels: dict[str, int] = field(default_factory=dict)
    sold_order_line_ids: set[str] = field(default_factory=set)
    requested_items: list[RequestedItem] = field(default_factory=list)
    financial_report: FinancialReport | None = None
