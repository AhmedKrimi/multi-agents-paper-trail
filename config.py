from inventory import PAPER_SUPPLIES
from sqlalchemy import create_engine
from items_model import RequestedItem
from pathlib import Path

# Abs path of the working directory
working_dir = Path(__file__).parent
# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")
# Extract available items
available_items = [paper["inventory_name"] for paper in PAPER_SUPPLIES]
products_price = {
    paper["inventory_name"]: paper["unit_price"] for paper in PAPER_SUPPLIES
}
# define response format from item_models
item_schema = RequestedItem.model_json_schema()
